"""Measure the layer profile of one model on one tier and print it as JSON (PLAN.md Phase 2 step 6).

Run inside the container of the tier, with this working tree mounted (the exact commands are
in experiments/profiles/README.md):

    python scripts/profile_layers.py device mobilenetv2 --checkpoint checkpoints/<name>.pt
    python scripts/profile_layers.py edge mobilenetv2 --checkpoint checkpoints/<name>.pt

The device tier is timed on one CPU thread with idle time before every run. The edge tier is
timed on the GPU with TF32 off, also with idle time before every run, as between two requests.
Without --checkpoint the weights are random, which does not change the times. If the checkpoint
has exit heads (scripts/train_exits.py), the device profile also has the times with the heads
running.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

from edgeinf.models.multi_exit import MODEL_NAMES, build_model, load_checkpoint
from edgeinf.profiler.layer_profile import profile_device, profile_edge, profile_exits

REQUESTS = 100  # timed runs per split point (PLAN.md: median of 100 runs after warm-up)
WARMUP = 10
GAP_MS = 50.0  # idle time before each device run, well above the 5 ms quota period
EDGE_GAP_MS = 200.0  # idle time before each edge run: the pause between requests at 5 per second
SEED = 0
CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tier", choices=("device", "edge"))
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--requests", type=int, default=REQUESTS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument(
        "--gap-ms", type=float, help=f"default: {GAP_MS} (device), {EDGE_GAP_MS} (edge)"
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    model = load_checkpoint(args.checkpoint) if args.checkpoint else build_model(args.model).eval()
    if model.name != args.model:
        parser.error(f"{args.checkpoint} holds {model.name}, not {args.model}")

    profile = {
        "model": args.model,
        "tier": args.tier,
        "n_blocks": model.num_blocks,
        "requests": args.requests,
        "warmup": args.warmup,
        "date": datetime.date.today().isoformat(),
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "cpu_max": CPU_MAX.read_text().strip() if CPU_MAX.exists() else None,
    }
    if args.tier == "device":
        torch.set_num_threads(1)  # a device has one core
        gap_ms = GAP_MS if args.gap_ms is None else args.gap_ms
        profile["gap_ms"] = gap_ms
        profile |= profile_device(model, args.requests, args.warmup, gap_ms, SEED)
        if model.exit_points:
            profile |= profile_exits(model, args.requests, args.warmup, gap_ms, SEED)
    else:
        torch.backends.cudnn.allow_tf32 = False  # the edge must compute what the device computes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        profile["device"] = torch.cuda.get_device_name() if device == "cuda" else "cpu"
        profile["tf32"] = False
        gap_ms = EDGE_GAP_MS if args.gap_ms is None else args.gap_ms
        profile["gap_ms"] = gap_ms
        profile |= profile_edge(model, args.requests, args.warmup, device, SEED, gap_ms)
    json.dump(profile, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
