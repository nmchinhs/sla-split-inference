"""Measure what JPEG costs the offload_jpeg variant and print it as JSON (PLAN.md Phase 2 step 7).

Run inside the container of the tier, with the exported frames mounted (the exact commands are in
experiments/jpeg/README.md):

    python scripts/profile_jpeg.py device
    python scripts/profile_jpeg.py edge mobilenetv2 --checkpoint checkpoints/<name>.pt

The device tier times encoding a frame, on one CPU thread with idle time before every run. The
edge tier measures accuracy and bytes for every frame of the split, times the edge's step, and
names the quality that the project's rule picks: the lowest that loses less than half a point of
accuracy against the raw frame.

Everything is measured on the validation split. The quality is a tuned value, so the test split
must not be used for it, and there is no option to do so.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

from edgeinf.data.imagenette import load_frames
from edgeinf.models.multi_exit import MODEL_NAMES, load_checkpoint
from edgeinf.profiler.jpeg_profile import choose_quality, quality_profile, time_device, time_edge

SPLIT = "val"
QUALITIES = (50, 75, 90, 95)  # PLAN.md Phase 2 step 7
MAX_DROP_POINTS = 0.5  # percentage points of accuracy a quality may lose (report log, D19)
REQUESTS = 100  # timed runs per quality, each on another frame
WARMUP = 10
GAP_MS = 50.0  # idle time before each device run, well above the 5 ms quota period
BATCH_SIZE = 64
CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def spread(frames: np.ndarray, count: int) -> torch.Tensor:
    """`count` frames evenly spaced over the split, which is stored class by class."""
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames))).astype(int)
    return torch.from_numpy(np.ascontiguousarray(frames[indices]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tier", choices=("device", "edge"))
    parser.add_argument("model", nargs="?", choices=MODEL_NAMES, help="edge tier only")
    parser.add_argument("--checkpoint", type=Path, help="edge tier only")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--requests", type=int, default=REQUESTS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--gap-ms", type=float, default=GAP_MS)
    args = parser.parse_args()
    if args.tier == "edge" and not (args.model and args.checkpoint):
        parser.error("the edge tier needs a model and --checkpoint")

    frames, labels = load_frames(args.frames, SPLIT)
    timed = spread(frames, args.requests + args.warmup)
    profile: dict = {
        "tier": args.tier,
        "split": SPLIT,
        "qualities": list(QUALITIES),
        "requests": args.requests,
        "warmup": args.warmup,
        "date": datetime.date.today().isoformat(),
        "torch": torch.__version__,
        "cpu_max": CPU_MAX.read_text().strip() if CPU_MAX.exists() else None,
    }
    if args.tier == "device":
        torch.set_num_threads(1)  # a device has one core
        profile["gap_ms"] = args.gap_ms
        profile |= time_device(timed, QUALITIES, args.requests, args.warmup, args.gap_ms)
    else:
        torch.backends.cudnn.allow_tf32 = False  # the edge must compute what the device computes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_checkpoint(args.checkpoint)
        if model.name != args.model:
            parser.error(f"{args.checkpoint} holds {model.name}, not {args.model}")
        profile |= {
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "tf32": False,
            "max_drop_points": MAX_DROP_POINTS,
        }
        everything = torch.from_numpy(np.array(frames))  # a copy in memory: the file is read-only
        measured = quality_profile(
            model, everything, labels.tolist(), QUALITIES, device, BATCH_SIZE
        )
        times = time_edge(model, timed, QUALITIES, args.requests, args.warmup, device)
        profile["frames"] = measured["frames"]
        profile["raw"] = measured["raw"] | times["raw"]
        profile["jpeg"] = {q: measured["jpeg"][q] | times["jpeg"][q] for q in QUALITIES}
        drops = {q: measured["jpeg"][q]["accuracy_drop_points"] for q in QUALITIES}
        try:
            profile["chosen_quality"] = choose_quality(drops, MAX_DROP_POINTS)
        except ValueError as error:
            profile["chosen_quality"] = None
            print(f"warning: {error}", file=sys.stderr)
    json.dump(profile, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
