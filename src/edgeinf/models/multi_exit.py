"""Multi-exit CNNs as an ordered list of blocks.

A model is `n` blocks, and split points are block boundaries: the device runs blocks [0, k) and
the edge runs blocks [k, n), so k = 0 offloads everything and k = n runs everything locally.

The architectures are torchvision's MobileNetV2 and ResNet-18, regrouped into blocks. They start
from random weights, never ImageNet-pretrained ones: Imagenette's val split (our test set) is
drawn from ImageNet's train images.

Exit heads are attached with `add_exits` (one `exit_head` per position) and trained by
scripts/train_exits.py with the backbone frozen. A model without them is a plain backbone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torchvision
from torch import nn

NUM_CLASSES = 10  # Imagenette
INPUT_SIZE = 224  # px, height and width of the model input
EXIT_KERNEL = 3  # px, kernel of the convolution in an exit head
PROBE_SIZE = 64  # px, side of the blank image that is run once to read the blocks' channel counts


class MultiExitModel(nn.Module):
    """A backbone as an ordered list of blocks, with optional exit heads.

    `blocks[i]` maps the output of block i-1 (the input image for i = 0) to the output of block
    i. The last block ends in the classifier and returns logits. `exits[str(i)]` is the exit head
    that reads the output of block i, so a device running blocks [0, k) evaluates the exits with
    i < k.

    Parameters live under `blocks.*` and `exits.*`, so a backbone checkpoint saved before any exit
    was added still loads afterwards (with `strict=False`, only `exits.*` is missing).
    """

    def __init__(self, name: str, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.name = name
        self.blocks = nn.ModuleList(blocks)
        self.exits = nn.ModuleDict()

    @property
    def num_blocks(self) -> int:
        """Number of blocks n. Valid split points are 0..n."""
        return len(self.blocks)

    @property
    def exit_points(self) -> list[int]:
        """Indices of the blocks followed by an exit head, in ascending order."""
        return sorted(int(key) for key in self.exits)

    def add_exit(self, after_block: int, head: nn.Module) -> None:
        """Attach `head` as the exit that reads the output of block `after_block`.

        The last block already returns the final output, so `after_block` must be in [0, n - 1).
        """
        if not 0 <= after_block < self.num_blocks - 1:
            raise ValueError(
                f"exit after block {after_block} is outside [0, {self.num_blocks - 1}) "
                f"for {self.name}"
            )
        if str(after_block) in self.exits:
            raise ValueError(f"{self.name} already has an exit after block {after_block}")
        self.exits[str(after_block)] = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every block and return the final logits, shape (batch, num_classes)."""
        for block in self.blocks:
            x = block(x)
        return x


def _classifier_block(classifier: nn.Module) -> nn.Sequential:
    """The last block: global average pool, flatten, then the classifier."""
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), classifier)


def _mobilenetv2_blocks(num_classes: int) -> list[nn.Module]:
    """20 blocks: stem conv, 17 inverted residuals, final 1x1 conv, classifier."""
    net = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
    return [*net.features, _classifier_block(net.classifier)]


def _resnet18_blocks(num_classes: int) -> list[nn.Module]:
    """10 blocks: stem (conv, batch norm, ReLU, max pool), 8 basic blocks, classifier."""
    net = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
    stages = [*net.layer1, *net.layer2, *net.layer3, *net.layer4]
    return [stem, *stages, _classifier_block(net.fc)]


_BUILDERS: dict[str, Callable[[int], list[nn.Module]]] = {
    "mobilenetv2": _mobilenetv2_blocks,
    "resnet18": _resnet18_blocks,
}
MODEL_NAMES = tuple(_BUILDERS)


def build_model(name: str, num_classes: int = NUM_CLASSES) -> MultiExitModel:
    """Build the backbone `name` (one of MODEL_NAMES) with random weights and no exits."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    return MultiExitModel(name, _BUILDERS[name](num_classes))


def exit_head(
    in_channels: int, pool: int, hidden: int, num_classes: int = NUM_CLASSES
) -> nn.Sequential:
    """An exit head: logits (batch, num_classes) from a block output with `in_channels` channels.

    The feature map is averaged down to `pool` x `pool`, whatever its size, so that every head
    costs about the same; then one 3x3 convolution to `hidden` channels, batch norm, ReLU, a
    global average and a linear classifier.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(pool),
        nn.Conv2d(in_channels, hidden, EXIT_KERNEL, padding=EXIT_KERNEL // 2, bias=False),
        nn.BatchNorm2d(hidden),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(hidden, num_classes),
    )


def block_channels(model: MultiExitModel) -> list[int]:
    """Channels of the output of every block but the last (which returns logits)."""
    was_training = model.training
    model.eval()  # so the probe does not touch the BatchNorm statistics
    parameter = next(model.parameters())
    x = torch.zeros(1, 3, PROBE_SIZE, PROBE_SIZE, device=parameter.device, dtype=parameter.dtype)
    channels = []
    with torch.no_grad():
        for block in model.blocks[:-1]:
            x = block(x)
            channels.append(x.shape[1])
    model.train(was_training)
    return channels


def add_exits(model: MultiExitModel, blocks: Sequence[int], pool: int, hidden: int) -> None:
    """Attach an `exit_head` with fresh weights after each of `blocks` (indices of blocks)."""
    channels = block_channels(model)
    for block in blocks:
        if not 0 <= block < len(channels):
            raise ValueError(
                f"exit after block {block} is outside [0, {len(channels)}) for {model.name}"
            )
        model.add_exit(block, exit_head(channels[block], pool, hidden))


def load_checkpoint(path: str | Path) -> MultiExitModel:
    """The model saved at `path`, on the CPU and in eval mode.

    The file is a backbone written by scripts/train_backbone.py, or a backbone with exit heads
    written by scripts/train_exits.py; the latter has an entry `exits` that says where the heads
    sit and how they are built (`blocks`, `pool`, `hidden`).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model", "state_dict"} <= set(checkpoint):
        raise ValueError(
            f"{path} is not a checkpoint written by scripts/train_backbone.py or train_exits.py"
        )
    model = build_model(checkpoint["model"])
    if exits := checkpoint.get("exits"):
        add_exits(model, exits["blocks"], exits["pool"], exits["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
