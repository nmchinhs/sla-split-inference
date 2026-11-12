"""Measure the exit profile of one model on the validation split (PLAN.md Phase 3 step 3).

    python scripts/profile_exits.py checkpoints/exits_mobilenetv2.pt

Reads the val frames written by scripts/export_frames.py, the frames a device would see, and
computes on the CPU in 32-bit floats, as a device does. Writes
experiments/exit_profiles/<model>.json; the meaning of its fields is in
src/edgeinf/profiler/exit_profile.py. There is no option for the test split on purpose:
thresholds are chosen from this file.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch

from edgeinf.data.imagenette import FRAMES_DIRNAME, load_frames
from edgeinf.models.multi_exit import load_checkpoint
from edgeinf.profiler.exit_profile import TAUS, exit_outputs, exit_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "experiments" / "exit_profiles"
SPLIT = "val"
BATCH_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="written by scripts/train_exits.py")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out", type=Path, help="default: experiments/exit_profiles/<model>.json")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    if not model.exit_points:
        parser.error(f"{args.checkpoint} has no exit heads")
    frames, labels = load_frames(args.data_root / FRAMES_DIRNAME, SPLIT)
    confidence, predicted, final = exit_outputs(model, frames, BATCH_SIZE)
    profile = {
        "model": model.name,
        "checkpoint": str(args.checkpoint),
        "split": SPLIT,
        "date": datetime.date.today().isoformat(),
        "torch": torch.__version__,
    } | exit_profile(confidence, predicted, final, labels, model.exit_points, TAUS)

    out = args.out or OUT_DIR / f"{model.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1) + "\n")

    print(f"{model.name}: {profile['samples']} {SPLIT} images, exits after blocks", end=" ")
    print(f"{profile['exit_blocks']}, final accuracy {profile['final_accuracy']:.4f}")
    print("head accuracy on all images:", [round(a, 4) for a in profile["head_accuracy"]])
    print("  tau | exit rate per exit | accuracy of each exit on what it takes | P_continue | acc.")
    for row in profile["by_tau"]:
        rates = " ".join(f"{r:.3f}" for r in row["exit_rate"])
        taken = " ".join("  -  " if a is None else f"{a:.3f}" for a in row["exit_accuracy"])
        print(
            f" {row['tau']:.2f} | {rates} | {taken} | "
            f"{row['p_continue'][-1]:.3f} | {row['accuracy'][-1]:.4f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
