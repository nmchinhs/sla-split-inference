"""Split execution (models/split.py): blocks [0, k) then [k, n) must equal the full forward pass."""

import pytest
import torch

from edgeinf.common.serialize import deserialize, serialize
from edgeinf.models.multi_exit import MODEL_NAMES, MultiExitModel, build_model
from edgeinf.models.split import run_blocks

SEED = 0
BYTES_PER_KB = 1024
# torch.allclose accepts differences up to 1e-8 whatever the scale, so the reference logits must
# be far larger than that for the comparisons to mean anything.
MIN_LOGIT = 1e-2
# Largest change of any logit caused by sending the split tensor as 16-bit floats, as a fraction
# of the largest logit. Measured on these models: 0.52% at worst (MobileNetV2, k = 0).
FP16_MAX_LOGIT_ERROR = 0.01
# From the feasibility study (Documents/report_log.md, D3), now checked on the real blocks and
# the real wire format: how many split points 0 < k < n send less than the raw 8-bit image when
# the tensor travels as 16-bit floats, and the size of the smallest such tensor.
SPLITS_SMALLER_THAN_IMAGE = {"mobilenetv2": 15, "resnet18": 4}
SMALLEST_SPLIT_KB = {"mobilenetv2": 15.3, "resnet18": 49.0}


@pytest.fixture(autouse=True)
def _inference_mode():
    with torch.inference_mode():
        yield


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request, settle_batchnorm) -> MultiExitModel:
    torch.manual_seed(SEED)
    model = build_model(request.param).eval()
    settle_batchnorm(model)
    return model


@pytest.fixture(scope="module")
def full(model, image) -> torch.Tensor:
    """The reference: logits of the unsplit forward pass."""
    with torch.inference_mode():
        logits = model(image)
    assert logits.abs().max() > MIN_LOGIT
    return logits


def test_split_equals_full_forward_for_every_k(model, image, full):
    n = model.num_blocks
    for k in range(n + 1):
        on_device = run_blocks(model, image, 0, k)
        on_edge = run_blocks(model, on_device, k, n)
        assert torch.allclose(on_edge, full), f"{model.name}: split at k={k} differs"


def test_split_through_the_wire_equals_full_forward_for_every_k(model, image, full):
    n = model.num_blocks
    # 0 <= k < n: at k = n nothing is sent.
    for k in range(n):
        received = deserialize(serialize(run_blocks(model, image, 0, k)))
        on_edge = run_blocks(model, received, k, n)
        assert torch.allclose(on_edge, full), f"{model.name}: split at k={k} differs"


def test_fp16_transfer_barely_changes_the_output(model, image, full):
    n = model.num_blocks
    for k in range(n):
        received = deserialize(serialize(run_blocks(model, image, 0, k), fp16=True))
        assert received.dtype == torch.float32
        error = (run_blocks(model, received, k, n) - full).abs().max() / full.abs().max()
        assert error < FP16_MAX_LOGIT_ERROR, f"{model.name}: k={k} changes a logit by {error:.1%}"


def test_empty_range_returns_the_input(model, image):
    for k in (0, model.num_blocks):
        assert run_blocks(model, image, k, k) is image


def test_blocks_do_not_modify_their_input(model, image):
    x = image
    for i in range(model.num_blocks):
        before = x.clone()
        y = run_blocks(model, x, i, i + 1)
        assert torch.equal(x, before), f"{model.name}: block {i} changed its input in place"
        x = y


def test_invalid_ranges_raise(model, image):
    n = model.num_blocks
    for start, end in ((-1, 1), (0, n + 1), (2, 1), (n + 1, n + 1)):
        with pytest.raises(ValueError):
            run_blocks(model, image, start, end)


def test_split_tensor_sizes_match_the_feasibility_study(model, image):
    n = model.num_blocks
    raw_image = torch.zeros(image.shape, dtype=torch.uint8)
    image_bytes = len(serialize(raw_image))
    sizes = [len(serialize(run_blocks(model, image, 0, k), fp16=True)) for k in range(1, n)]

    assert sum(size < image_bytes for size in sizes) == SPLITS_SMALLER_THAN_IMAGE[model.name]
    assert round(min(sizes) / BYTES_PER_KB, 1) == SMALLEST_SPLIT_KB[model.name]
