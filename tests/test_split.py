"""Split execution (models/split.py): blocks [0, k) then [k, n) must equal the full forward pass."""

import pytest
import torch

from edgeinf.common.serialize import deserialize, serialize, serialize_jpeg
from edgeinf.data.imagenette import normalise
from edgeinf.models.multi_exit import MODEL_NAMES, MultiExitModel, build_model
from edgeinf.models.split import pack, run_blocks, run_head, run_tail, top1

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
JPEG_QUALITY = 90
HEADER_BYTES_4D = 22  # of a serialised 4-D tensor
FP16_BYTES = 2


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


def test_head_is_the_frame_itself_at_k_0_and_the_normalised_blocks_after(model, frame):
    assert run_head(model, frame, 0) is frame
    x = normalise(frame)
    for k in range(1, model.num_blocks + 1):
        assert torch.equal(run_head(model, frame, k), run_blocks(model, x, 0, k))


def test_head_then_tail_is_the_whole_model_for_every_k(model, frame):
    # This is also the device's fallback: it finishes with run_tail on the tensor it has.
    full = model(normalise(frame))
    assert full.abs().max() > MIN_LOGIT
    for k in range(model.num_blocks + 1):
        logits = run_tail(model, run_head(model, frame, k), k)
        assert torch.allclose(logits, full), f"{model.name}: k={k} differs"


def test_tail_normalises_only_the_raw_frame(model, frame):
    n = model.num_blocks
    assert torch.equal(run_tail(model, frame, 0), run_blocks(model, normalise(frame), 0, n))
    logits = run_head(model, frame, n)
    assert run_tail(model, logits, n) is logits


def test_top1_is_the_largest_softmax_and_its_index():
    label, confidence = top1(torch.tensor([[1.0, 3.0, 2.0]]))
    assert isinstance(label, int) and isinstance(confidence, float)
    assert label == 1
    assert confidence == pytest.approx(torch.tensor([1.0, 3.0, 2.0]).softmax(0)[1].item())


def test_pack_sends_the_raw_frame_at_k_0_and_16_bit_floats_after(model, frame):
    assert pack(frame, 0) == serialize(frame)
    for k in range(1, model.num_blocks):
        x = run_head(model, frame, k)
        data = pack(x, k)
        assert data == serialize(x, fp16=True)
        assert len(data) == HEADER_BYTES_4D + FP16_BYTES * x.numel()


def test_pack_sends_a_jpeg_at_k_0_when_a_quality_is_given(frame):
    assert pack(frame, 0, jpeg_quality=JPEG_QUALITY) == serialize_jpeg(frame, JPEG_QUALITY)
    assert len(pack(frame, 0, jpeg_quality=JPEG_QUALITY)) < len(pack(frame, 0))


def test_pack_refuses_a_jpeg_of_anything_but_the_frame(model, frame):
    with pytest.raises(ValueError, match="k = 0"):
        pack(run_head(model, frame, 1), 1, jpeg_quality=JPEG_QUALITY)


def test_what_is_packed_gives_the_whole_models_label_at_the_other_end(model, frame):
    label, confidence = top1(model(normalise(frame)))
    for k in range(model.num_blocks):
        received = deserialize(pack(run_head(model, frame, k), k))
        got_label, got_confidence = top1(run_tail(model, received, k))
        assert got_label == label, f"{model.name}: k={k}"
        assert got_confidence == pytest.approx(confidence, rel=FP16_MAX_LOGIT_ERROR)
