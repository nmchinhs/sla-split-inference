"""Feasibility check for split inference: does any split point beat local/offload?

Profiles MobileNetV2 block by block (latency in ms, output size in bytes) at several
input resolutions, then estimates end-to-end latency for every split point k under
simple assumptions. Used to choose the dataset / input size (see Documents/report_log.md).

Usage:
  taskset -c 4 python scripts/feasibility_check.py profile 1 dev.json     # "device" core
  taskset -c 0,1 python scripts/feasibility_check.py profile 2 edge.json  # "edge" cores
  python scripts/feasibility_check.py analyze dev.json edge.json

Weights are random: latency and tensor sizes do not depend on weight values.
"""

import json
import statistics
import sys
import time

import torch
import torchvision

RESOLUTIONS = (32, 224, 320, 448, 512)
WARMUP_RUNS = 10
TIMED_RUNS = 30
NUM_CLASSES = 10
RTT_MS = 20.0
BANDWIDTHS_MBPS = (1, 2, 5, 10, 20, 50)
DEVICE_SLOWDOWNS = (10, 20)  # device = N x slower than one free laptop core
P_CONTINUE = (1.0, 0.5, 0.3)  # fraction of inputs NOT exiting on the device
FIRST_EXIT_BLOCK = 4  # assume early exits are available from this split point on
FP16_BYTES = 2


def build_blocks(res: int) -> list[torch.nn.Module]:
    """MobileNetV2 as an ordered list of blocks (19 feature blocks + classifier head)."""
    model = torchvision.models.mobilenet_v2(num_classes=NUM_CLASSES)
    if res == 32:  # common CIFAR adaptation: remove two early stride-2 downsamplings
        model.features[0][0].stride = (1, 1)
        model.features[2].conv[1][0].stride = (1, 1)
    head = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), model.classifier
    )
    return [b.eval() for b in [*model.features, head]]


def median_ms(fn) -> float:
    """Median wall time of fn() in ms after warm-up."""
    for _ in range(WARMUP_RUNS):
        fn()
    times = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


@torch.inference_mode()
def profile(threads: int) -> dict:
    """Per-block latency (ms) and output element count for each resolution."""
    torch.set_num_threads(threads)
    out = {}
    for res in RESOLUTIONS:
        x = torch.randn(1, 3, res, res)
        rows = []
        for block in build_blocks(res):
            ms = median_ms(lambda block=block, x=x: block(x))
            x = block(x)
            rows.append({"ms": ms, "out_numel": x.numel()})
        out[str(res)] = {"blocks": rows, "input_uint8_bytes": 3 * res * res}
    return out


def latency_ms(k: int, dev_ms, edge_ms, sent, bw_mbps: float, p_cont: float) -> float:
    """Expected latency (ms) of split point k; k = n means fully local."""
    n = len(dev_ms)
    if k == n:
        return sum(dev_ms)
    p = p_cont if k >= FIRST_EXIT_BLOCK else 1.0
    transfer_ms = sent[k] * 8 / (bw_mbps * 1e3)
    return sum(dev_ms[:k]) + p * (transfer_ms + RTT_MS + sum(edge_ms[k:]))


def analyze(dev: dict, edge: dict) -> None:
    for res, d in dev.items():
        blocks = d["blocks"]
        n = len(blocks)
        in_bytes = d["input_uint8_bytes"]
        # k = 0 sends the raw uint8 image; 0 < k < n sends the fp16 output of block k-1
        sent = [in_bytes] + [b["out_numel"] * FP16_BYTES for b in blocks[:-1]]
        edge_ms = [b["ms"] for b in edge[res]["blocks"]]
        smaller = [k for k in range(1, n) if sent[k] < in_bytes]
        one_core = sum(b["ms"] for b in blocks)
        print(
            f"\n=== {res}x{res}: raw image {in_bytes / 1024:.1f} KB, "
            f"smallest split tensor {min(sent[1:]) / 1024:.1f} KB, "
            f"full model {one_core:.1f} ms (1 core), edge {sum(edge_ms):.1f} ms (2 cores)"
        )
        print(
            f"    split points sending less than the raw image: {len(smaller)} of {n - 1}"
        )
        for slow in DEVICE_SLOWDOWNS:
            dev_ms = [b["ms"] * slow for b in blocks]
            for p_cont in P_CONTINUE:
                cells = []
                for bw in BANDWIDTHS_MBPS:
                    lats = [
                        latency_ms(k, dev_ms, edge_ms, sent, bw, p_cont)
                        for k in range(n + 1)
                    ]
                    best = min(range(n + 1), key=lats.__getitem__)
                    name = (
                        "offload"
                        if best == 0
                        else "local"
                        if best == n
                        else f"split{best}"
                    )
                    cells.append(f"{bw}M:{name} {lats[best]:.0f}")
                print(
                    f"    dev {slow}x (local {sum(dev_ms):.0f} ms), P_cont={p_cont}: "
                    + " | ".join(cells)
                )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "profile":
        with open(sys.argv[3], "w") as f:
            json.dump(profile(int(sys.argv[2])), f)
    elif cmd == "analyze":
        with open(sys.argv[2]) as f_dev, open(sys.argv[3]) as f_edge:
            analyze(json.load(f_dev), json.load(f_edge))
    else:
        sys.exit(__doc__)
