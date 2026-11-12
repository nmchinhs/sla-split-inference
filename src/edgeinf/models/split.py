"""Split execution: the device runs blocks [0, k), the edge runs blocks [k, n).

`run_blocks` runs a range of blocks. The four steps below are what one request is made of, and
the device runtime, the edge server and the layer profiler all use them, so that what is profiled
is what runs:

    run_head   the device's part: frame -> what crosses the split point
    pack       that tensor as bytes for the edge
    run_tail   the rest: what crossed the split point -> logits (on the edge, or on the device
               when it falls back)
    top1       logits -> label and confidence

With exit heads the device's part is `run_head_exits`: after every block that has a head it
checks the head's confidence against tau, and answers at once if it is high enough.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from edgeinf.common.serialize import serialize, serialize_jpeg
from edgeinf.data.imagenette import normalise
from edgeinf.models.multi_exit import MultiExitModel

TAU_NEVER = 1.0  # at this threshold (and above) no exit is taken, and no exit head is run


class Exit(NamedTuple):
    """An answer given by an exit head on the device."""

    block: int  # the head reads the output of this block
    label: int  # predicted class index
    confidence: float  # largest softmax of the head, 0..1


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


def exit_check(model: MultiExitModel, x: torch.Tensor, block: int) -> tuple[int, float]:
    """Label and confidence of the exit head after `block` for `x`, the output of that block."""
    return top1(model.exits[str(block)](x))


def _walk(
    model: MultiExitModel, frame: torch.Tensor, k: int, take: Callable[[int, float], bool]
) -> tuple[torch.Tensor, Exit | None]:
    """`run_head` with every exit before k evaluated; stops at the first one `take` accepts.

    `take(block, confidence)` decides. Returns the output of the last block that ran and the
    exit that was taken, or None if the frame got through to split point k.
    """
    if not 0 <= k <= model.num_blocks:
        raise ValueError(f"split point {k} is outside [0, {model.num_blocks}] for {model.name}")
    if k == 0:
        return frame, None
    x = normalise(frame)
    for block in range(k):
        x = model.blocks[block](x)
        if str(block) in model.exits:
            label, confidence = exit_check(model, x, block)
            if take(block, confidence):
                return x, Exit(block, label, confidence)
    return x, None


def run_head_exits(
    model: MultiExitModel, frame: torch.Tensor, k: int, tau: float
) -> tuple[torch.Tensor, Exit | None]:
    """The device's part for one 8-bit frame with early exits: blocks [0, k) and the exits in it.

    The exits after blocks i < k are checked in order, and the first whose confidence is
    >= `tau` answers: the result is then (output of that block, the Exit). Otherwise it is
    (what crosses split point k, None), as from `run_head`. With tau = 1.0 the heads are not run
    at all, so that "never exit" costs nothing and cannot be defeated by a confidence that rounds
    to exactly 1.0.
    """
    if tau >= TAU_NEVER:
        return run_head(model, frame, k), None
    return _walk(model, frame, k, lambda _block, confidence: confidence >= tau)


def run_head_forced(
    model: MultiExitModel, frame: torch.Tensor, k: int, exit_block: int | None
) -> tuple[torch.Tensor, Exit | None]:
    """`run_head_exits` with the outcome fixed, for timing: every exit before k is evaluated,
    and the one after `exit_block` is taken whatever its confidence (None: no exit is taken)."""
    return _walk(model, frame, k, lambda block, _confidence: block == exit_block)
