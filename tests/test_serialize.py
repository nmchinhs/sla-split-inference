"""Tensor <-> bytes for the device-to-edge transfer (common/serialize.py)."""

import math
import struct
import warnings

import pytest
import torch

from edgeinf.common.serialize import deserialize, serialize

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


def _random(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    if dtype.is_floating_point:
        return (torch.randn(shape, generator=generator) * ACTIVATION_SCALE).to(dtype)
    info = torch.iinfo(dtype)
    low, high = max(info.min, -(2**31)), min(info.max, 2**31 - 1)
    return torch.randint(low, high + 1, shape, generator=generator).to(dtype)


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
