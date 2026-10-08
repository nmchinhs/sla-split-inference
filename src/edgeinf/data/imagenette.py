"""Imagenette at 224x224, as three splits.

The dataset's own "train" folder is divided into train and val (stratified by class, fixed by a
seed); its "val" folder is the test set. Models and exits are trained on train, thresholds and
exit profiles are calibrated on val, and results are reported on test only.

The 320 px version of the dataset is used: its images have a shorter side of 320 px, which is
enough for 224x224 crops.

A device does not read dataset files. It gets frames: the val or test images already resized and
cropped, as 8-bit pixels, written once by `export_frames` (scripts/export_frames.py). Decoding and
resizing a photo is the camera's work and must not count as inference time.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from edgeinf.models.multi_exit import INPUT_SIZE

DIRNAME = "imagenette2-320"  # folder the archive unpacks to
SIZE = "320px"  # torchvision's name for that version
# Folder (WordNet id) and name of each class, in label order.
CLASSES = (
    ("n01440764", "tench"),
    ("n02102040", "English springer"),
    ("n02979186", "cassette player"),
    ("n03000684", "chain saw"),
    ("n03028079", "church"),
    ("n03394916", "French horn"),
    ("n03417042", "garbage truck"),
    ("n03425413", "gas pump"),
    ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
)
WNIDS = tuple(wnid for wnid, _ in CLASSES)
CLASS_NAMES = tuple(name for _, name in CLASSES)

# Per-channel mean and standard deviation of ImageNet images, the usual input normalisation.
# These are statistics of pixels, not pretrained weights.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PIXEL_MAX = 255.0  # largest value of an 8-bit pixel
RESIZE = 256  # px, shorter side before the centre crop of val and test images
MIN_CROP_AREA = 0.35  # smallest random crop of a training image, as a fraction of its area
FRAMES_DIRNAME = "frames"  # folder under the data root that holds the exported frames
FRAMES_FILE = "{split}_frames.npy"  # uint8, shape (N, 3, 224, 224)
LABELS_FILE = "{split}_labels.npy"  # int64, shape (N,)
FRAME_SPLITS = ("val", "test")  # the splits whose images are the same on every read

_CROP = [transforms.Resize(RESIZE), transforms.CenterCrop(INPUT_SIZE)]
_NORMALISE = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(MIN_CROP_AREA, 1.0)),
        transforms.RandomHorizontalFlip(),
        *_NORMALISE,
    ]
)
EVAL_TRANSFORM = transforms.Compose([*_CROP, *_NORMALISE])
FRAME_TRANSFORM = transforms.Compose([*_CROP, transforms.PILToTensor()])  # 8-bit, not normalised


def stratified_split(
    labels: Sequence[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices 0..len(labels)-1 into (train, val), with `val_fraction` of every class in val.

    The result depends only on `labels`, `val_fraction` and `seed`. Both lists are sorted.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        n_val = round(val_fraction * len(indices))
        val += indices[:n_val]
        train += indices[n_val:]
    return sorted(train), sorted(val)


@dataclass
class Splits:
    """The three datasets. Each item is (image tensor 3x224x224, label 0..9)."""

    train: Dataset
    val: Dataset
    test: Dataset
    _files: dict[str, list[str]]
    _labels: dict[str, list[int]]

    def files(self, split: str) -> list[str]:
        """Image paths of "train", "val" or "test", in dataset order."""
        return self._files[split]

    def labels(self, split: str) -> list[int]:
        """Class indices of "train", "val" or "test", in dataset order."""
        return self._labels[split]


def download(root: str | Path) -> None:
    """Download and unpack the 320 px Imagenette archive (about 330 MB) into `root`."""
    datasets.Imagenette(str(root), split="train", size=SIZE, download=True)


def load_splits(root: str | Path, seed: int, val_fraction: float) -> Splits:
    """Return train, val and test from the dataset unpacked under `root`.

    Only training images are augmented (random crop and flip); val and test images are resized
    and centre-cropped, so they are the same on every read.
    """
    folder = Path(root) / DIRNAME
    if not (folder / "train").is_dir() or not (folder / "val").is_dir():
        raise FileNotFoundError(
            f"{folder} not found: download Imagenette first (scripts/train_backbone.py --download)"
        )
    augmented = datasets.ImageFolder(folder / "train", TRAIN_TRANSFORM)
    plain = datasets.ImageFolder(folder / "train", EVAL_TRANSFORM)
    test = datasets.ImageFolder(folder / "val", EVAL_TRANSFORM)
    if tuple(augmented.classes) != WNIDS or tuple(test.classes) != WNIDS:
        raise ValueError(f"unexpected class folders under {folder}")

    train_idx, val_idx = stratified_split(augmented.targets, val_fraction, seed)
    paths = [path for path, _ in augmented.samples]
    return Splits(
        train=Subset(augmented, train_idx),
        val=Subset(plain, val_idx),
        test=test,
        _files={
            "train": [paths[i] for i in train_idx],
            "val": [paths[i] for i in val_idx],
            "test": [path for path, _ in test.samples],
        },
        _labels={
            "train": [augmented.targets[i] for i in train_idx],
            "val": [augmented.targets[i] for i in val_idx],
            "test": list(test.targets),
        },
    )


def export_frames(splits: Splits, split: str, directory: str | Path) -> Path:
    """Write the images of "val" or "test" as 8-bit frames, and their labels, into `directory`.

    A frame is the image resized and centre-cropped as in EVAL_TRANSFORM but left as 8-bit
    pixels, so `normalise(frame)` is exactly what the models were trained and evaluated on.
    Returns the path of the frames file (150,528 bytes per image).
    """
    if split not in FRAME_SPLITS:
        raise ValueError(f"frames exist for {FRAME_SPLITS}, not for {split!r}")
    files = splits.files(split)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FRAMES_FILE.format(split=split)
    shape = (len(files), 3, INPUT_SIZE, INPUT_SIZE)
    frames = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    for index, file in enumerate(files):
        with Image.open(file) as image:
            frames[index] = FRAME_TRANSFORM(image.convert("RGB")).numpy()
    frames.flush()
    np.save(directory / LABELS_FILE.format(split=split), np.asarray(splits.labels(split)))
    return path


def load_frames(directory: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Frames and labels written by `export_frames`: uint8 (N, 3, 224, 224) and int64 (N,).

    The frames are memory-mapped and read-only, so only the ones that are used are read: the
    test split is 591 MB, more than a device container has.
    """
    directory = Path(directory)
    try:
        frames = np.load(directory / FRAMES_FILE.format(split=split), mmap_mode="r")
        labels = np.load(directory / LABELS_FILE.format(split=split))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"no {split} frames in {directory}: write them first (scripts/export_frames.py)"
        ) from error
    if frames.dtype != np.uint8 or frames.shape != (len(labels), 3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError(f"{directory}: {split} frames have shape {frames.shape}, {frames.dtype}")
    return frames, labels


def normalise(image: torch.Tensor) -> torch.Tensor:
    """Turn 8-bit images (N, 3, H, W), values 0..255, into the float input the models expect.

    This is what ToTensor and Normalize do in the dataset transforms. The device applies it to a
    frame before block 0; the edge applies it when it receives the raw image (k = 0).
    """
    mean = torch.tensor(MEAN, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=image.device).view(1, 3, 1, 1)
    return (image.float() / PIXEL_MAX - mean) / std
