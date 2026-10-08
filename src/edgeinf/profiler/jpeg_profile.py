"""JPEG for the offload_jpeg variant: what a quality costs in accuracy, in bytes and in time.

`quality_profile` sends every frame of a split through the wire format as a JPEG, classifies what
arrives, and compares with sending the raw 8-bit frame. `choose_quality` applies the project's
rule to the result. `time_device` and `time_edge` time the two ends of an offloaded request, as
the layer profile does for k = 0: the device's step is encoding the frame, the edge's step is
everything from the received bytes to the answer.

The quality is a tuned value, so all of this is measured on the validation split.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Mapping, Sequence

import torch

from edgeinf.common.serialize import deserialize
from edgeinf.data.imagenette import normalise
from edgeinf.models.multi_exit import MultiExitModel
from edgeinf.models.split import pack
from edgeinf.profiler.layer_profile import edge_step, median_and_p95, time_calls

PERCENT = 100.0


@torch.inference_mode()
def predictions(
    model: MultiExitModel,
    frames: torch.Tensor,
    quality: int | None,
    device: str,
    batch_size: int,
) -> tuple[list[int], list[int]]:
    """Predicted label of every frame after it travelled as a JPEG, and the bytes sent for each.

    `frames` is uint8, shape (N, 3, H, W). With `quality` None the frames travel as raw 8-bit
    pixels, which is the reference.
    """
    model = model.to(device)
    labels: list[int] = []
    sizes: list[int] = []
    for start in range(0, len(frames), batch_size):
        received = []
        for index in range(start, min(start + batch_size, len(frames))):
            data = pack(frames[index : index + 1], 0, quality)
            sizes.append(len(data))
            received.append(deserialize(data))
        logits = model(normalise(torch.cat(received).to(device)))
        labels += logits.argmax(1).tolist()
    return labels, sizes


def _accuracy(predicted: Sequence[int], truth: Sequence[int]) -> float:
    return sum(p == t for p, t in zip(predicted, truth, strict=True)) / len(truth)


def quality_profile(
    model: MultiExitModel,
    frames: torch.Tensor,
    labels: Sequence[int],
    qualities: Sequence[int],
    device: str,
    batch_size: int,
) -> dict:
    """Accuracy and bytes sent for the raw frame and for a JPEG of each quality.

    Returns {"frames": N, "raw": {"accuracy", "bytes"}, "jpeg": {quality: {...}}}. Accuracy is a
    fraction; `accuracy_drop_points` is how many percentage points a quality loses against the
    raw frame; `labels_changed` counts the frames whose predicted label differs from the raw
    frame's, in either direction; the bytes include the 3-byte header.
    """
    truth = list(labels)
    raw_labels, raw_sizes = predictions(model, frames, None, device, batch_size)
    raw_accuracy = _accuracy(raw_labels, truth)
    profile: dict = {
        "frames": len(truth),
        "raw": {"accuracy": raw_accuracy, "bytes": raw_sizes[0]},
        "jpeg": {},
    }
    for quality in qualities:
        got, sizes = predictions(model, frames, quality, device, batch_size)
        accuracy = _accuracy(got, truth)
        median, p95 = median_and_p95(sizes)
        profile["jpeg"][quality] = {
            "accuracy": accuracy,
            "accuracy_drop_points": PERCENT * (raw_accuracy - accuracy),
            "labels_changed": sum(a != b for a, b in zip(got, raw_labels, strict=True)),
            "bytes_mean": statistics.mean(sizes),
            "bytes_median": median,
            "bytes_p95": p95,
            "bytes_min": min(sizes),
            "bytes_max": max(sizes),
        }
    return profile


def choose_quality(drop_points: Mapping[int, float], max_drop_points: float) -> int:
    """The lowest quality that loses less than `max_drop_points` percentage points of accuracy.

    `drop_points` maps each quality to its loss against the raw frame. Raises ValueError if no
    quality qualifies.
    """
    good = [quality for quality, drop in drop_points.items() if drop < max_drop_points]
    if not good:
        raise ValueError(
            f"no quality loses less than {max_drop_points} points of accuracy: {dict(drop_points)}"
        )
    return min(good)


def time_device(
    frames: torch.Tensor, qualities: Sequence[int], requests: int, warmup: int, gap_ms: float
) -> dict:
    """Median and p95 time in ms of the device's step at k = 0, raw and for each JPEG quality.

    The step is turning the frame into the bytes to send. Runs cycle through `frames`, because
    how long a JPEG takes depends on the image. `gap_ms` of idle time precedes every run, as
    between requests at a fixed rate.
    """

    def measure(quality: int | None) -> dict[str, float]:
        position = itertools.count()

        def step() -> None:
            index = next(position) % len(frames)
            pack(frames[index : index + 1], 0, quality)

        median, p95 = median_and_p95(time_calls(step, requests, warmup, gap_ms))
        return {"device_ms": median, "device_p95_ms": p95}

    return {"raw": measure(None), "jpeg": {quality: measure(quality) for quality in qualities}}


@torch.inference_mode()
def time_edge(
    model: MultiExitModel,
    frames: torch.Tensor,
    qualities: Sequence[int],
    requests: int,
    warmup: int,
    device: str,
) -> dict:
    """Median and p95 time in ms of the edge's step at k = 0, raw and for each JPEG quality.

    The step is everything from the received bytes to label and confidence, so for a JPEG it
    includes decoding it. The bytes are produced first and not timed.
    """
    model = model.to(device)

    def measure(quality: int | None) -> dict[str, float]:
        payloads = [pack(frames[index : index + 1], 0, quality) for index in range(len(frames))]
        position = itertools.count()

        def step() -> None:
            edge_step(model, payloads[next(position) % len(payloads)], 0, device)

        median, p95 = median_and_p95(time_calls(step, requests, warmup, 0.0))
        return {"edge_ms": median, "edge_p95_ms": p95}

    return {"raw": measure(None), "jpeg": {quality: measure(quality) for quality in qualities}}
