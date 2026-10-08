"""Tensor <-> bytes for the device-to-edge transfer (common/serialize.py)."""

import io
import math
import struct
import warnings

import pytest
import torch
from PIL import Image

from edgeinf.common.serialize import deserialize, serialize, serialize_jpeg

SEED = 0
IMAGE_SHAPE = (1, 3, 224, 224)  # model input
SPLIT_SHAPE = (1, 160, 7, 7)  # smallest MobileNetV2 split tensor at 224x224
SHAPES = ((), (5,), (2, 0, 3), SPLIT_SHAPE, IMAGE_SHAPE)
DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
)
HEADER_BYTES_4D = 22  # magic 2, version 1, two dtype codes 2, ndim 1, four uint32 dims 16
FP32_BYTES = 4
FP16_BYTES = 2
FP16_MAX = 65504.0  # largest finite 16-bit float
FP16_SMALLEST = 2.0**-24  # smallest positive 16-bit float (about 6e-8)
FP16_MAX_REL_ERROR = 2.0**-11  # round to nearest with an 11-bit significand
FP16_SUBNORMAL_ERROR = 2.0**-25  # half the spacing of 16-bit floats below 2**-14
ACTIVATION_SCALE = 6.0  # same order as ReLU6 outputs
PHOTO_SHAPE = (1, 3, 96, 160)  # not square, so swapped axes cannot pass
JPEG_QUALITIES = (50, 75, 90, 95)
JPEG_HEADER = b"ET" + bytes([2])  # magic and format version 2: the rest is a JPEG file
JPEG_START, JPEG_END = b"\xff\xd8", b"\xff\xd9"  # markers that open and close a JPEG file
# Mean absolute error in grey levels (0..255) of the smooth test image after a round trip at
# quality 90. Measured: 0.8, and 1.1 in the worst channel. Two swapped channels give 57.
JPEG_MAX_MEAN_ERROR = 2.0
JPEG_MIN_SAVING = 5  # the test image must shrink at least this many times at every quality


def _random(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    if dtype.is_floating_point:
        return (torch.randn(shape, generator=generator) * ACTIVATION_SCALE).to(dtype)
    info = torch.iinfo(dtype)
    low, high = max(info.min, -(2**31)), min(info.max, 2**31 - 1)
    return torch.randint(low, high + 1, shape, generator=generator).to(dtype)


def _photo(shape: tuple[int, ...] = PHOTO_SHAPE) -> torch.Tensor:
    """A smooth 8-bit RGB image with three different channels, as JPEG is made for."""
    _, _, height, width = shape
    y = torch.linspace(0, 1, height).view(height, 1).expand(height, width)
    x = torch.linspace(0, 1, width).view(1, width).expand(height, width)
    wave = 0.5 + 0.5 * torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * y)
    return (255 * torch.stack([x, y, wave])).round().to(torch.uint8).unsqueeze(0)


def _mean_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_round_trip_is_exact_and_keeps_dtype(dtype):
    tensor = _random(SPLIT_SHAPE, dtype)
    restored = deserialize(serialize(tensor))
    assert restored.dtype == dtype
    assert torch.equal(restored, tensor)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_round_trip_keeps_shape(shape):
    tensor = _random(shape)
    restored = deserialize(serialize(tensor))
    assert restored.shape == tensor.shape
    assert torch.equal(restored, tensor)


def test_fp16_round_trip_restores_float32_at_half_precision():
    tensor = _random(SPLIT_SHAPE)
    restored = deserialize(serialize(tensor, fp16=True))
    assert restored.dtype == torch.float32
    assert restored.shape == tensor.shape
    assert torch.equal(restored, tensor.half().float())
    bound = FP16_MAX_REL_ERROR * tensor.abs() + FP16_SUBNORMAL_ERROR
    assert ((restored - tensor).abs() <= bound).all()


def test_size_is_header_plus_payload_and_fp16_halves_the_payload():
    tensor = _random(SPLIT_SHAPE)
    numel = math.prod(SPLIT_SHAPE)
    assert len(serialize(tensor)) == HEADER_BYTES_4D + numel * FP32_BYTES
    assert len(serialize(tensor, fp16=True)) == HEADER_BYTES_4D + numel * FP16_BYTES


def test_fp16_leaves_integer_tensors_unchanged():
    image = _random(IMAGE_SHAPE, torch.uint8)
    data = serialize(image, fp16=True)
    assert len(data) == HEADER_BYTES_4D + math.prod(IMAGE_SHAPE)
    restored = deserialize(data)
    assert restored.dtype == torch.uint8
    assert torch.equal(restored, image)


def test_fp16_loses_values_outside_the_half_precision_range():
    # Too large becomes inf, too small becomes 0: the transfer only works for activations of
    # ordinary size, which a trained model with BatchNorm has and an untrained one may not.
    tensor = torch.tensor([FP16_MAX, 2 * FP16_MAX, -2 * FP16_MAX, FP16_SMALLEST, FP16_SMALLEST / 4])
    restored = deserialize(serialize(tensor, fp16=True))
    assert restored.tolist() == [FP16_MAX, math.inf, -math.inf, FP16_SMALLEST, 0.0]


def test_wire_format_is_pinned():
    # Device and edge images are built separately, so the exact bytes are a contract.
    tensor = torch.tensor([[1.0, -2.0]])
    shape = struct.pack("<2I", 1, 2)
    float32, float16 = 6, 5  # dtype codes
    assert serialize(tensor) == (
        b"ET" + bytes([1, float32, float32, 2]) + shape + struct.pack("<2f", 1.0, -2.0)
    )
    assert serialize(tensor, fp16=True) == (
        b"ET" + bytes([1, float32, float16, 2]) + shape + struct.pack("<2e", 1.0, -2.0)
    )


def test_serialize_accepts_non_contiguous_tensors():
    tensor = _random(SPLIT_SHAPE).permute(0, 2, 3, 1)
    assert not tensor.is_contiguous()
    assert torch.equal(deserialize(serialize(tensor)), tensor)


def test_serialize_accepts_tensors_that_require_grad():
    tensor = _random(SPLIT_SHAPE).requires_grad_()
    restored = deserialize(serialize(tensor))
    assert not restored.requires_grad
    assert torch.equal(restored, tensor.detach())


def test_serialize_rejects_unsupported_dtypes():
    with pytest.raises(ValueError, match="complex64"):
        serialize(torch.zeros(2, dtype=torch.complex64))


def test_deserialize_returns_an_independent_writable_tensor():
    tensor = _random(SPLIT_SHAPE)
    data = serialize(tensor)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = deserialize(data)
        restored.add_(1.0)
    assert torch.equal(deserialize(data), tensor)


_VALID = b"ET" + bytes([1, 6, 6, 2]) + struct.pack("<2I", 1, 2) + struct.pack("<2f", 1.0, -2.0)
MALFORMED = {
    "empty": b"",
    "truncated header": _VALID[:5],
    "truncated shape": _VALID[:9],
    "wrong magic": b"XX" + _VALID[2:],
    "unknown version": _VALID[:2] + b"\xff" + _VALID[3:],
    "unknown dtype": _VALID[:3] + b"\xff" + _VALID[4:],
    "unknown payload dtype": _VALID[:4] + b"\xff" + _VALID[5:],
    "truncated payload": _VALID[:-1],
    "trailing bytes": _VALID + b"\x00",
}


def test_the_malformed_cases_start_from_a_valid_encoding():
    assert torch.equal(deserialize(_VALID), torch.tensor([[1.0, -2.0]]))


@pytest.mark.parametrize("data", MALFORMED.values(), ids=MALFORMED.keys())
def test_deserialize_rejects_malformed_bytes(data):
    with pytest.raises(ValueError):
        deserialize(data)


def test_jpeg_round_trip_keeps_shape_and_dtype_and_stays_close():
    image = _photo()
    restored = deserialize(serialize_jpeg(image, quality=90))
    assert restored.dtype == torch.uint8
    assert restored.shape == image.shape
    assert _mean_error(restored, image) < JPEG_MAX_MEAN_ERROR
    # Each channel on its own, so that a swap of two channels cannot hide in the mean.
    for channel in range(3):
        assert _mean_error(restored[:, channel], image[:, channel]) < JPEG_MAX_MEAN_ERROR


def test_jpeg_is_much_smaller_than_the_raw_image_and_grows_with_quality():
    image = _photo()
    sizes = [len(serialize_jpeg(image, quality)) for quality in JPEG_QUALITIES]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert max(sizes) * JPEG_MIN_SAVING < len(serialize(image))


def test_a_higher_jpeg_quality_is_closer_to_the_original():
    image = _photo()
    errors = [_mean_error(deserialize(serialize_jpeg(image, q)), image) for q in JPEG_QUALITIES]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]


def test_jpeg_wire_format_is_pinned():
    # Version 2 of the format: the three header bytes, then one complete JPEG file.
    data = serialize_jpeg(_photo(), quality=75)
    assert data[: len(JPEG_HEADER)] == JPEG_HEADER
    jpeg = data[len(JPEG_HEADER) :]
    assert jpeg[:2] == JPEG_START and jpeg[-2:] == JPEG_END
    with Image.open(io.BytesIO(jpeg)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.size == (PHOTO_SHAPE[3], PHOTO_SHAPE[2])  # PIL gives (width, height)
        assert decoded.mode == "RGB"


def test_jpeg_encoding_is_deterministic():
    image = _photo()
    assert serialize_jpeg(image, quality=75) == serialize_jpeg(image, quality=75)


@pytest.mark.parametrize(
    "image",
    [
        _photo().float(),  # not 8-bit
        _photo()[0],  # no batch dimension
        _photo().repeat(2, 1, 1, 1),  # two images
        _photo()[:, :1],  # one channel
    ],
    ids=["float", "3-D", "batch of 2", "1 channel"],
)
def test_serialize_jpeg_rejects_what_is_not_one_8_bit_rgb_image(image):
    with pytest.raises(ValueError):
        serialize_jpeg(image, quality=75)


@pytest.mark.parametrize("quality", [0, 101, -1])
def test_serialize_jpeg_rejects_a_quality_outside_1_to_100(quality):
    with pytest.raises(ValueError, match="quality"):
        serialize_jpeg(_photo(), quality)


def test_deserialize_jpeg_returns_an_independent_writable_tensor():
    image = _photo()
    data = serialize_jpeg(image, quality=90)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = deserialize(data)
        expected = restored.clone()
        restored.add_(1)
    assert torch.equal(deserialize(data), expected)


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    return buffer.getvalue()


_VALID_JPEG = serialize_jpeg(_photo(), quality=75)
MALFORMED_JPEG = {
    "header only": JPEG_HEADER,
    "not a jpeg": JPEG_HEADER + bytes(range(256)),
    "truncated jpeg": _VALID_JPEG[: len(_VALID_JPEG) // 2],
    "another image format": JPEG_HEADER + _png(),
    "jpeg without the header": _VALID_JPEG[len(JPEG_HEADER) :],
}


@pytest.mark.parametrize("data", MALFORMED_JPEG.values(), ids=MALFORMED_JPEG.keys())
def test_deserialize_rejects_malformed_jpeg(data):
    with pytest.raises(ValueError):
        deserialize(data)
