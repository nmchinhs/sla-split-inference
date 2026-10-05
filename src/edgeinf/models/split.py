"""Run a contiguous range of a model's blocks: the device runs [0, k), the edge runs [k, n)."""

from __future__ import annotations

import torch

from edgeinf.models.multi_exit import MultiExitModel


def run_blocks(model: MultiExitModel, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Run blocks [start, end) on `x` and return the output of block end - 1.

    `x` must be the output of block start - 1, or the input image when start = 0. An empty range
    (start = end) returns `x` itself. The caller puts the model in eval mode and disables
    gradients.
    """
    if not 0 <= start <= end <= model.num_blocks:
        raise ValueError(
            f"block range [{start}, {end}) is outside [0, {model.num_blocks}] for {model.name}"
        )
    for i in range(start, end):
        x = model.blocks[i](x)
    return x
