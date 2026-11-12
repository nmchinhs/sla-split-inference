"""Train the exit heads of one model on a frozen, already trained backbone (PLAN.md Phase 3).

    python scripts/train_exits.py experiments/configs/exits_mobilenetv2.yaml

Only the heads learn. The backbone stays in eval mode and gets no gradient, so its weights and
BatchNorm statistics, and with them the final output, are exactly those of the backbone
checkpoint; the script checks this before it writes anything. Every head has its own
cross-entropy loss, without label smoothing: an exit is taken when its confidence passes a
threshold, and smoothed labels would keep the confidence away from 1.

Trains on the train split and measures every exit on the val split after each epoch. The test
split is not touched. Writes checkpoints/<name>.pt (backbone and heads, read by
`load_checkpoint`) and one row per epoch to results/training/<name>.csv, where <name> is the
config file's name.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from edgeinf.data.imagenette import load_splits
from edgeinf.models.multi_exit import MultiExitModel, add_exits, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
LOG_DIR = REPO_ROOT / "results" / "training"

CONFIG_KEYS = {
    "seed",
    "backbone",
    "exits",
    "head_pool",
    "head_hidden",
    "data_root",
    "val_fraction",
    "epochs",
    "batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "workers",
    "device",
    "amp",
}
BYTES_PER_MB = 2**20


def log_fields(exits: list[int]) -> list[str]:
    """Columns of the per-epoch log for a model with exits after the blocks `exits`."""
    return [
        "epoch",
        "train_loss",  # sum of the heads' cross-entropy losses, mean over the epoch
        *(f"val_accuracy_exit{block}" for block in exits),  # each head on all val images
        "val_accuracy_final",
        "lr",
        "epoch_s",
        "peak_vram_mb",
    ]


def forward_all(model: MultiExitModel, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Logits of every exit head (in block order) and of the final output for a batch.

    The blocks run without gradient; the heads are ordinary modules on top of their outputs.
    """
    exits = model.exit_points
    exit_logits = []
    for block in range(model.num_blocks):
        with torch.no_grad():
            x = model.blocks[block](x)
        if block in exits:
            exit_logits.append(model.exits[str(block)](x))
    return exit_logits, x


@torch.no_grad()
def evaluate(model: MultiExitModel, loader: DataLoader, device: str, amp: bool) -> list[float]:
    """Top-1 accuracy (0..1) on `loader` of every exit head, then of the final output."""
    model.eval()
    correct = [0] * (len(model.exit_points) + 1)
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device, enabled=amp):
            exit_logits, final = forward_all(model, images)
        for i, logits in enumerate([*exit_logits, final]):
            correct[i] += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return [c / count for c in correct]


def train(config: dict, checkpoint_path: Path, log_path: Path) -> None:
    """Train the heads as described by `config` and write the checkpoint and the per-epoch log."""
    if unknown := set(config) - CONFIG_KEYS:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if missing := CONFIG_KEYS - set(config):
        raise ValueError(f"missing config keys: {sorted(missing)}")
    seed, device = config["seed"], config["device"]
    amp = bool(config["amp"]) and device == "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    backbone_path = Path(config["backbone"])
    if not backbone_path.is_absolute():
        backbone_path = REPO_ROOT / backbone_path
    model = load_checkpoint(backbone_path)
    if model.exit_points:
        raise ValueError(f"{backbone_path} already has exits after blocks {model.exit_points}")
    backbone_state = {key: value.clone() for key, value in model.state_dict().items()}
    exits = sorted(config["exits"])
    add_exits(model, exits, config["head_pool"], config["head_hidden"])
    model.to(device)
    for parameter in model.blocks.parameters():
        parameter.requires_grad_(False)

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

    optimiser = torch.optim.SGD(
        model.exits.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, config["epochs"] * len(train_loader)
    )
    scaler = torch.amp.GradScaler(device, enabled=amp)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fields = log_fields(exits)
    accuracies: list[float] = []
    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            start = time.perf_counter()
            model.eval()  # the backbone's BatchNorm and dropout stay as in inference
            model.exits.train()
            loss_sum, count = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device, enabled=amp):
                    exit_logits, _ = forward_all(model, images)
                    loss = sum(
                        nn.functional.cross_entropy(logits, labels) for logits in exit_logits
                    )
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item() * labels.numel()
                count += labels.numel()
            accuracies = evaluate(model, val_loader, device, amp)
            row = dict.fromkeys(fields)
            row |= {
                "epoch": epoch,
                "train_loss": round(loss_sum / count, 4),
                "val_accuracy_final": round(accuracies[-1], 4),
                "lr": round(scheduler.get_last_lr()[0], 6),
                "epoch_s": round(time.perf_counter() - start, 1),
                "peak_vram_mb": (
                    round(torch.cuda.max_memory_allocated() / BYTES_PER_MB)
                    if device == "cuda"
                    else 0
                ),
            }
            for block, accuracy in zip(exits, accuracies, strict=False):
                row[f"val_accuracy_exit{block}"] = round(accuracy, 4)
            writer.writerow(row)
            log_file.flush()
            per_exit = "  ".join(
                f"exit{block} {accuracy:.3f}"
                for block, accuracy in zip(exits, accuracies, strict=False)
            )
            print(
                f"epoch {epoch:3d}/{config['epochs']}  train loss {row['train_loss']:.3f}  "
                f"val acc: {per_exit}  final {accuracies[-1]:.3f}  {row['epoch_s']:.0f} s  "
                f"{row['peak_vram_mb']} MB",
                flush=True,
            )

    state = {key: value.cpu() for key, value in model.state_dict().items()}
    changed = [key for key, value in backbone_state.items() if not torch.equal(state[key], value)]
    if changed:
        raise RuntimeError(f"the frozen backbone changed during training: {changed[:5]}")
    torch.save(
        {
            "model": model.name,
            "state_dict": state,
            "exits": {
                "blocks": exits,
                "pool": config["head_pool"],
                "hidden": config["head_hidden"],
            },
            "epochs": config["epochs"],
            "val_accuracy": accuracies[-1],
            "exit_val_accuracy": accuracies[:-1],
            "config": dict(config),
        },
        checkpoint_path,
    )
    print(f"wrote {checkpoint_path} (final val accuracy {accuracies[-1]:.4f}) and {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML file describing the run")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    name = args.config.stem
    train(config, CHECKPOINT_DIR / f"{name}.pt", LOG_DIR / f"{name}.csv")


if __name__ == "__main__":
    main()
