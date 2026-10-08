"""Split execution: the device runs blocks [0, k), the edge runs blocks [k, n).

`run_blocks` runs a range of blocks. The four steps below are what one request is made of, and
the device runtime, the edge server and the layer profiler all use them, so that what is profiled
is what runs:

    run_head   the device's part: frame -> what crosses the split point
    pack       that tensor as bytes for the edge
    run_tail   the rest: what crossed the split point -> logits (on the edge, or on the device
               when it falls back)
    top1       logits -> label and confidence
"""

from __future__ import annotations

import torch

from edgeinf.common.serialize import serialize, serialize_jpeg
from edgeinf.data.imagenette import normalise
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


def run_head(model: MultiExitModel, frame: torch.Tensor, k: int) -> torch.Tensor:
    """What crosses split point k for one 8-bit frame, shape (1, 3, H, W).

    For k = 0 that is the frame itself. Otherwise the frame is normalised and blocks [0, k) run,
    which gives the output of block k - 1, and the logits when k = n.
    """
    if k == 0:
        return frame
    return run_blocks(model, normalise(frame), 0, k)


def run_tail(model: MultiExitModel, x: torch.Tensor, k: int) -> torch.Tensor:
    """Logits from `x`, which is what crossed split point k: blocks [k, n).

    The raw frame (k = 0) is normalised first. With k = n nothing is left to run and `x` is
    returned, since it already holds the logits.
    """
    if k == 0:
        x = normalise(x)
    return run_blocks(model, x, k, model.num_blocks)


def pack(x: torch.Tensor, k: int, jpeg_quality: int | None = None) -> bytes:
    """Bytes to send for `x`, which is what crosses split point k.

    The frame (k = 0) goes as raw 8-bit pixels, or as a JPEG of quality `jpeg_quality` if one is
    given. The output of a block (k > 0) goes as 16-bit floats.
    """
    if jpeg_quality is not None:
        if k != 0:
            raise ValueError(
                f"only the frame can be sent as a JPEG, that is with k = 0, not k = {k}"
            )
        return serialize_jpeg(x, jpeg_quality)
    return serialize(x, fp16=k > 0)


def top1(logits: torch.Tensor) -> tuple[int, float]:
    """Predicted class index and its confidence (the largest softmax) for logits of shape (1, C).

    Reading the values waits for a GPU to finish.
    """
    confidence, label = logits.softmax(1).max(1)
    return int(label), float(confidence)
