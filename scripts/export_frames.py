"""Write the val and test images as 8-bit 224x224 frames for the device runtime.

    python scripts/export_frames.py experiments/configs/train_mobilenetv2.yaml

The data root, the seed and the val fraction are read from a training config, so that "val" here
is the val split the backbones were trained with. Writes <data_root>/frames/{val,test}_frames.npy
(uint8, N x 3 x 224 x 224) and {val,test}_labels.npy. The device containers mount that folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from edgeinf.data.imagenette import FRAME_SPLITS, FRAMES_DIRNAME, export_frames, load_splits

BYTES_PER_MB = 10**6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="a training config (data_root, seed, split)")
    parser.add_argument("splits", nargs="*", help=f"some of {FRAME_SPLITS}; default: all")
    args = parser.parse_args()
    if unknown := set(args.splits) - set(FRAME_SPLITS):
        parser.error(f"no frames for {sorted(unknown)}; choose from {FRAME_SPLITS}")

    config = yaml.safe_load(args.config.read_text())
    splits = load_splits(config["data_root"], config["seed"], config["val_fraction"])
    for split in args.splits or FRAME_SPLITS:
        path = export_frames(splits, split, Path(config["data_root"]) / FRAMES_DIRNAME)
        size_mb = path.stat().st_size / BYTES_PER_MB
        print(f"wrote {path}: {len(splits.files(split))} frames, {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
