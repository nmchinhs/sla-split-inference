"""Multi-exit CNNs as an ordered list of blocks.

A model is `n` blocks, and split points are block boundaries: the device runs blocks [0, k) and
the edge runs blocks [k, n), so k = 0 offloads everything and k = n runs everything locally.

The architectures are torchvision's MobileNetV2 and ResNet-18, regrouped into blocks. They start
from random weights, never ImageNet-pretrained ones: Imagenette's val split (our test set) is
drawn from ImageNet's train images.

Exit heads are attached in Phase 3 with `MultiExitModel.add_exit`. Until then `exits` is empty
and the model is a plain backbone.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision
from torch import nn

NUM_CLASSES = 10  # Imagenette
INPUT_SIZE = 224  # px, height and width of the model input


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
