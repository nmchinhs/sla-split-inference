"""Train a plain backbone on Imagenette from random weights (PLAN.md Phase 2 step 4).

    python scripts/train_backbone.py experiments/configs/train_mobilenetv2.yaml [--download]

Trains on the train split and measures accuracy on the val split after every epoch. The test
split is not touched here. Writes the weights after the last epoch to checkpoints/<name>.pt and
one row per epoch to results/training/<name>.csv, where <name> is the config file's name.

`--download` fetches the dataset (about 330 MB) into the config's `data_root` first.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from edgeinf.data.imagenette import download, load_splits
from edgeinf.models.multi_exit import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR = REPO_ROOT / "results" / "training"

CONFIG_KEYS = {
    "seed",
    "model",
    "data_root",
    "val_fraction",
    "epochs",
    "batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "label_smoothing",
    "warmup_epochs",
    "workers",
    "device",
    "amp",
}
LOG_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "lr",
    "epoch_s",
    "peak_vram_mb",
)
BYTES_PER_MB = 2**20


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning rate at `step` as a fraction of the peak: linear warm-up, then cosine to zero."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, amp: bool) -> tuple[float, float]:
    """Mean loss and top-1 accuracy (0..1) of `model` on `loader`."""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device, enabled=amp):
            logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits.float(), labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return loss_sum / count, correct / count


def train(config: dict, checkpoint_path: Path, log_path: Path) -> None:
    """Train as described by `config` and write the checkpoint and the per-epoch log."""
    if unknown := set(config) - CONFIG_KEYS:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if missing := CONFIG_KEYS - set(config):
        raise ValueError(f"missing config keys: {sorted(missing)}")
    seed, device = config["seed"], config["device"]
    amp = bool(config["amp"]) and device == "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    splits = load_splits(config["data_root"], seed, config["val_fraction"])
    loader_args = {
        "batch_size": config["batch_size"],
        "num_workers": config["workers"],
        "pin_memory": device == "cuda",
        "persistent_workers": config["workers"] > 0,
    }
    train_loader = DataLoader(
        splits.train,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_args,
    )
    val_loader = DataLoader(splits.val, shuffle=False, **loader_args)

    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )
    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: lr_factor(step, warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler(device, enabled=amp)
    # With mixed precision the scaler may skip the very first optimiser step, and PyTorch then
    # warns that the schedule was stepped first. One skipped step of the warm-up does not matter.
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)`")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    val_accuracy = 0.0
    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            start = time.perf_counter()
            model.train()
            loss_sum, correct, count = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device, enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item() * labels.numel()
                correct += (logits.argmax(1) == labels).sum().item()
                count += labels.numel()
            val_loss, val_accuracy = evaluate(model, val_loader, device, amp)
            row = {
                "epoch": epoch,
                "train_loss": round(loss_sum / count, 4),
                "train_accuracy": round(correct / count, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_accuracy, 4),
                "lr": round(scheduler.get_last_lr()[0], 6),
                "epoch_s": round(time.perf_counter() - start, 1),
                "peak_vram_mb": (
                    round(torch.cuda.max_memory_allocated() / BYTES_PER_MB)
                    if device == "cuda"
                    else 0
                ),
            }
            writer.writerow(row)
            log_file.flush()
            print(
                f"epoch {epoch:3d}/{config['epochs']}  train loss {row['train_loss']:.3f} "
                f"acc {row['train_accuracy']:.3f}  val loss {row['val_loss']:.3f} "
                f"acc {row['val_accuracy']:.3f}  {row['epoch_s']:.0f} s  "
                f"{row['peak_vram_mb']} MB",
                flush=True,
            )

    torch.save(
        {
            "model": config["model"],
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "epochs": config["epochs"],
            "val_accuracy": val_accuracy,
            "config": dict(config),
        },
        checkpoint_path,
    )
    print(f"wrote {checkpoint_path} (val accuracy {val_accuracy:.4f}) and {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML file describing the run")
    parser.add_argument("--download", action="store_true", help="fetch the dataset first")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.download:
        download(config["data_root"])
    name = args.config.stem
    train(config, CHECKPOINT_DIR / f"{name}.pt", LOG_DIR / f"{name}.csv")


if __name__ == "__main__":
    main()
