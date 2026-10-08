"""Layer profile: device time, edge time and bytes sent for every split point k.

Under a CPU quota the times of single blocks do not add up (experiments/calibration.md), so the
device side is measured as a whole for every k: everything the device does to a frame before it
sends, or answers when k = n. The edge side is likewise everything from receiving the bytes to
having the answer.

    device_ms[k]  k = 0..n    normalise the frame, run blocks [0, k), serialise the result
                              (k = 0: only serialise the raw frame; k = n: no serialising)
    bytes[k]      k = 0..n    size of what is sent: the raw 8-bit frame at 0, the output of
                              block k-1 as 16-bit floats for 0 < k < n, nothing at n
    edge_ms[k]    k = 0..n-1  decode, move to the edge's device, (k = 0: normalise,) run blocks
                              [k, n), read out label and confidence
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from edgeinf.common.serialize import deserialize, serialize
from edgeinf.data.imagenette import normalise
from edgeinf.models.multi_exit import INPUT_SIZE, MultiExitModel
from edgeinf.models.split import run_blocks

P95 = 0.95
MS_PER_S = 1e3
PIXEL_VALUES = 256


def raw_image(seed: int) -> torch.Tensor:
    """A random 8-bit frame, shape (1, 3, 224, 224). Timing does not depend on its content."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
    return torch.randint(0, PIXEL_VALUES, shape, generator=generator).to(torch.uint8)


def _time(fn: Callable[[], object], requests: int, warmup: int, gap_ms: float) -> list[float]:
    """Wall time in ms of `requests` calls of fn after `warmup` calls, idle `gap_ms` before each."""
    times = []
    for i in range(warmup + requests):
        if gap_ms:
            time.sleep(gap_ms / MS_PER_S)
        start = time.perf_counter()
        fn()
        if i >= warmup:
            times.append((time.perf_counter() - start) * MS_PER_S)
    return times


def _median_and_p95(times: list[float]) -> tuple[float, float]:
    ordered = sorted(times)
    return statistics.median(ordered), ordered[max(round(P95 * len(ordered)), 1) - 1]


def device_step(model: MultiExitModel, frame: torch.Tensor, k: int) -> bytes | int:
    """What the device does to one 8-bit frame at split point k: returns the bytes to send, or
    the predicted label when k = n."""
    if k == 0:
        return serialize(frame)
    out = run_blocks(model, normalise(frame), 0, k)
    if k == model.num_blocks:
        return int(out.argmax(1))
    return serialize(out, fp16=True)


def edge_step(model: MultiExitModel, data: bytes, k: int, device: str) -> tuple[int, float]:
    """What the edge does with the bytes of split point k: returns (label, confidence)."""
    x = deserialize(data).to(device)
    if k == 0:
        x = normalise(x)
    confidence, label = run_blocks(model, x, k, model.num_blocks).softmax(1).max(1)
    return int(label), float(confidence)  # reading the values waits for the GPU to finish


@torch.inference_mode()
def profile_device(
    model: MultiExitModel, requests: int, warmup: int, gap_ms: float, seed: int
) -> dict[str, list[float]]:
    """Median and p95 device time in ms, and bytes sent, for k = 0..n. See the module docstring.

    `gap_ms` of idle time precedes every timed run, as between requests that arrive at a fixed
    rate, so every run starts with a fresh CPU quota.
    """
    frame = raw_image(seed)
    medians, p95s, sizes = [], [], []
    for k in range(model.num_blocks + 1):
        sent = device_step(model, frame, k)
        sizes.append(len(sent) if isinstance(sent, bytes) else 0)
        median, p95 = _median_and_p95(
            _time(lambda k=k: device_step(model, frame, k), requests, warmup, gap_ms)
        )
        medians.append(median)
        p95s.append(p95)
    return {"device_ms": medians, "device_p95_ms": p95s, "bytes": sizes}


@torch.inference_mode()
def profile_edge(
    model: MultiExitModel, requests: int, warmup: int, device: str, seed: int
) -> dict[str, list[float]]:
    """Median and p95 edge time in ms for k = 0..n-1, with the model on `device`.

    The bytes for each k are produced on the CPU first, as a device would, and not timed.
    """
    frame = raw_image(seed)
    cpu_model = model.to("cpu")
    payloads = [device_step(cpu_model, frame, k) for k in range(model.num_blocks)]
    model = model.to(device)
    medians, p95s = [], []
    for k, data in enumerate(payloads):
        median, p95 = _median_and_p95(
            _time(lambda k=k, data=data: edge_step(model, data, k, device), requests, warmup, 0.0)
        )
        medians.append(median)
        p95s.append(p95)
    return {"edge_ms": medians, "edge_p95_ms": p95s}
