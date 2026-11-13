"""Predicted latency and accuracy of a plan (k, tau), from the profiles and the link's state.

A request takes one of several paths:

    exit j    the exit head after block b_j < k is the first that is sure enough. Share of the
              inputs: exit_rate[j] of the exit profile at tau. Time: exit_ms[j] of the layer
              profile. Nothing is sent.
    the rest  no exit before k answers; share P_continue(k, tau). Time:
                  T_slow = T_dev(k) + bytes(k) / bw + rtt + T_edge(k) * (1 + queue_len)
              and just T_dev(n) when k = n, where nothing is sent.

    mean  =  sum over the paths of share * time
    p95   =  the time below which 95% of the requests are answered, the paths taken in order
             of their time

The paths are in that order by construction: an exit before k answers before the device has run
all of [0, k). If the paths before one answer a share `before` of the requests and that path a
share `s`, with before < 0.95 <= before + s, the 95th percentile of all requests is that path's
quantile q = (0.95 - before) / s. Of a path only the median and the 95th percentile are
measured, so its time is taken as linear in the quantile through those two points:

    time(q) = median + (q - 0.5) / 0.45 * (p95 - median)

A plan with one path has q = 0.95 and gets that path's p95. With a slow path of share s >= 5%
the p95 is that path's quantile 1 - 0.05 / s. The paths are treated as if their times did not
overlap.

Until 2026-11-13 the rule took the whole path's p95 instead of time(q). That was up to 55% too
slow when nearly all requests exit (report log, D29).

T_dev(k) is `device_exits_ms[k]` when heads ran on the way (tau < 1 and an exit before k), and
`device_ms[k]` otherwise. For the p95 the device and edge terms are their own 95th percentiles;
the network term is the same in both, except that a JPEG's size varies with the image.

Accuracy is that of the exit profile, measured on the validation split: the exits' answers where
they answer, the final output for the rest.

All times in ms, sizes in bytes, bandwidth in Mbps, accuracies and shares as fractions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path as FilePath

from edgeinf.device.bandwidth import transfer_ms

EXPERIMENTS_DIR = FilePath(__file__).resolve().parents[3] / "experiments"
P95 = 0.95
MEDIAN = 0.5
TAU_NEVER = 1.0  # at this threshold no exit head is run (models/split.py)
SHARE_EPSILON = 1e-9  # shares are sums of floats


@dataclass(frozen=True)
class Link:
    """The state a prediction is made for: uplink bandwidth, round-trip time, edge queue."""

    bw_mbps: float
    rtt_ms: float
    queue_len: int = 0  # Infer requests at the edge ahead of this one


@dataclass(frozen=True)
class Path:
    """One way a request of a plan can go."""

    name: str  # "exit <block>", "edge" or "local"
    share: float  # fraction of the inputs that take it
    ms: float  # median time
    p95_ms: float


@dataclass(frozen=True)
class Prediction:
    mean_ms: float
    p95_ms: float
    accuracy: float
    p_continue: float  # share of the inputs that no exit answers
    paths: tuple[Path, ...]  # in order of time; paths that no input takes are left out


@dataclass(frozen=True)
class Profiles:
    """The measured profiles of one model, as the scripts wrote them.

    device  layer profile of the device tier (profiler/layer_profile.py), with the exit times
    edge    layer profile of the edge tier
    exits   exit profile (profiler/exit_profile.py)
    jpeg    by quality: device_ms, device_p95_ms, edge_ms, edge_p95_ms, bytes_mean, bytes_p95,
            accuracy (profiler/jpeg_profile.py); empty if the frame is never sent as a JPEG
    """

    model: str
    n_blocks: int
    device: dict
    edge: dict
    exits: dict
    jpeg: dict[int, dict]


def load_profiles(model: str, root: FilePath = EXPERIMENTS_DIR) -> Profiles:
    """Read the profiles of `model` from the folder `root` (experiments/ of the repository)."""

    def read(*parts: str) -> dict:
        return json.loads(root.joinpath(*parts).read_text())

    device = read("profiles", f"{model}_device.json")
    jpeg_device = read("jpeg", "device.json")["jpeg"]
    jpeg_edge = read("jpeg", f"{model}_edge.json")["jpeg"]
    return Profiles(
        model=model,
        n_blocks=device["n_blocks"],
        device=device,
        edge=read("profiles", f"{model}_edge.json"),
        exits=read("exit_profiles", f"{model}.json"),
        jpeg={int(quality): jpeg_device[quality] | jpeg_edge[quality] for quality in jpeg_edge},
    )


def _exit_row(profiles: Profiles, tau: float) -> dict:
    for row in profiles.exits["by_tau"]:
        if row["tau"] == tau:
            return row
    known = [row["tau"] for row in profiles.exits["by_tau"]]
    raise ValueError(f"the exit profile has no tau = {tau}; it has {known}")


def _p95_ms(paths: tuple[Path, ...]) -> float:
    """The time in ms below which 95% of the requests are answered (see the module docstring)."""
    before = 0.0
    for path in paths[:-1]:
        if before + path.share >= P95 - SHARE_EPSILON:
            break
        before += path.share
    else:
        path = paths[-1]
    quantile = min((P95 - before) / path.share, 1.0)
    return path.ms + (quantile - MEDIAN) / (P95 - MEDIAN) * (path.p95_ms - path.ms)


def predict(
    profiles: Profiles, k: int, tau: float, link: Link, jpeg_quality: int | None = None
) -> Prediction:
    """Latency and accuracy of serving with split point `k` and threshold `tau` over `link`.

    With `jpeg_quality` the frame is sent as a JPEG of that quality, which needs k = 0. `tau`
    must be a threshold of the exit profile, or 1.0. See the module docstring for the model.
    """
    n = profiles.n_blocks
    if not 0 <= k <= n:
        raise ValueError(f"split point {k} is outside [0, {n}] for {profiles.model}")
    if jpeg_quality is not None and k != 0:
        raise ValueError(f"only the frame can be sent as a JPEG, that is with k = 0, not k = {k}")
    if jpeg_quality is not None and jpeg_quality not in profiles.jpeg:
        raise ValueError(
            f"no JPEG profile for quality {jpeg_quality}; have {sorted(profiles.jpeg)}"
        )
    if k < n and link.bw_mbps <= 0:
        raise ValueError(f"a plan that sends needs a bandwidth above 0, got {link.bw_mbps}")

    device = profiles.device
    exit_blocks = device.get("exit_blocks", [])
    in_use = 0 if tau >= TAU_NEVER else sum(block < k for block in exit_blocks)
    paths = []
    p_continue, accuracy = 1.0, profiles.exits["final_accuracy"]
    if in_use:
        row = _exit_row(profiles, tau)
        p_continue, accuracy = row["p_continue"][in_use - 1], row["accuracy"][in_use - 1]
        exit_ms, exit_p95_ms = device["exit_ms"], device["exit_p95_ms"]
        paths = [
            Path(f"exit {exit_blocks[j]}", row["exit_rate"][j], exit_ms[j], exit_p95_ms[j])
            for j in range(in_use)
        ]
    elif tau < TAU_NEVER:
        _exit_row(profiles, tau)  # refuse a threshold the profile does not have

    heads = "device_exits_ms" if in_use else "device_ms"
    slow_ms, slow_p95_ms = device[heads][k], device[heads.replace("_ms", "_p95_ms")][k]
    if k < n:
        size, size_p95 = device["bytes"][k], device["bytes"][k]
        edge_ms, edge_p95_ms = profiles.edge["edge_ms"][k], profiles.edge["edge_p95_ms"][k]
        if jpeg_quality is not None:
            jpeg = profiles.jpeg[jpeg_quality]
            slow_ms, slow_p95_ms = jpeg["device_ms"], jpeg["device_p95_ms"]
            size, size_p95 = jpeg["bytes_mean"], jpeg["bytes_p95"]
            edge_ms, edge_p95_ms = jpeg["edge_ms"], jpeg["edge_p95_ms"]
            accuracy = jpeg["accuracy"]
        waiting = 1 + link.queue_len
        slow_ms += transfer_ms(size, link.bw_mbps) + link.rtt_ms + edge_ms * waiting
        slow_p95_ms += transfer_ms(size_p95, link.bw_mbps) + link.rtt_ms + edge_p95_ms * waiting
    paths.append(Path("edge" if k < n else "local", p_continue, slow_ms, slow_p95_ms))

    paths = tuple(path for path in paths if path.share > 0)
    return Prediction(
        mean_ms=sum(path.share * path.ms for path in paths),
        p95_ms=_p95_ms(paths),
        accuracy=accuracy,
        p_continue=p_continue,
        paths=paths,
    )
