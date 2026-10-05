"""Tensor <-> bytes for the device-to-edge transfer.

Wire format, all integers little-endian:

    offset   bytes   field
    0        2       magic b"ET"
    2        1       format version
    3        1       dtype code of the tensor, which is what `deserialize` returns
    4        1       dtype code of the payload (float16 when sent in half precision)
    5        1       number of dimensions d
    6        4 * d   shape, one uint32 per dimension
    6 + 4d   rest    payload: the elements in C order, little-endian

The header makes the bytes self-describing, so the receiver needs nothing else to decode them and
the encoding can change without touching proto/inference.proto.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import torch

MAGIC = b"ET"
VERSION = 1
_PREFIX = struct.Struct("<2sBBBB")  # magic, version, dtype code, payload dtype code, ndim
_SHAPE_FORMAT = "<{}I"  # one uint32 per dimension

# A dtype's wire code is its position here, so only ever append.
_DTYPES: tuple[tuple[torch.dtype, np.dtype], ...] = (
    (torch.uint8, np.dtype(np.uint8)),
    (torch.int8, np.dtype(np.int8)),
    (torch.int16, np.dtype(np.int16)),
    (torch.int32, np.dtype(np.int32)),
    (torch.int64, np.dtype(np.int64)),
    (torch.float16, np.dtype(np.float16)),
    (torch.float32, np.dtype(np.float32)),
    (torch.float64, np.dtype(np.float64)),
)
_CODES = {dtype: code for code, (dtype, _) in enumerate(_DTYPES)}


def _on_wire(native: np.dtype) -> np.dtype:
    """The little-endian form of a NumPy dtype."""
    return native.newbyteorder("<")


def serialize(tensor: torch.Tensor, fp16: bool = False) -> bytes:
    """Encode `tensor` as self-describing bytes: 6 + 4 * ndim header bytes, then the payload.

    With `fp16`, floating-point tensors travel as 16-bit floats (2 bytes per element instead of
    4 for float32) and `deserialize` casts them back to their own dtype. Values are rounded to
    the nearest 16-bit float: the relative error is at most 2**-11 (0.05%) for magnitudes from
    2**-14 (about 6e-5) to 65504. Larger magnitudes become inf and magnitudes below about 3e-8
    become 0, so this suits activations of ordinary size, as a trained model has. Integer
    tensors are sent unchanged.

    Raises ValueError for a dtype that has no wire code.
    """
    dtype = tensor.dtype
    if dtype not in _CODES:
        raise ValueError(f"cannot serialize dtype {dtype}")
    payload_dtype = torch.float16 if fp16 and dtype.is_floating_point else dtype
    payload_code = _CODES[payload_dtype]

    header = _PREFIX.pack(MAGIC, VERSION, _CODES[dtype], payload_code, tensor.ndim)
    shape = struct.pack(_SHAPE_FORMAT.format(tensor.ndim), *tensor.shape)
    array = tensor.detach().to("cpu", payload_dtype).numpy()
    return header + shape + array.astype(_on_wire(_DTYPES[payload_code][1]), copy=False).tobytes()


def deserialize(data: bytes) -> torch.Tensor:
    """Decode bytes written by `serialize` into a CPU tensor with the original shape and dtype.

    Raises ValueError if `data` is not one complete, well-formed encoding.
    """
    if len(data) < _PREFIX.size:
        raise ValueError(f"{len(data)} bytes is shorter than the {_PREFIX.size}-byte header")
    magic, version, code, payload_code, ndim = _PREFIX.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"not a version-{VERSION} tensor encoding")
    if code >= len(_DTYPES) or payload_code >= len(_DTYPES):
        raise ValueError(f"unknown dtype code {max(code, payload_code)}")

    shape_struct = struct.Struct(_SHAPE_FORMAT.format(ndim))
    payload_start = _PREFIX.size + shape_struct.size
    if len(data) < payload_start:
        raise ValueError(f"{len(data)} bytes is too short for a {ndim}-dimensional shape")
    shape = shape_struct.unpack_from(data, _PREFIX.size)

    native = _DTYPES[code][1]
    payload = _on_wire(_DTYPES[payload_code][1])
    count = math.prod(shape)
    if len(data) - payload_start != count * payload.itemsize:
        raise ValueError(
            f"shape {shape} needs {count * payload.itemsize} payload bytes, "
            f"got {len(data) - payload_start}"
        )
    array = np.frombuffer(data, dtype=payload, count=count, offset=payload_start)
    # astype copies, so the tensor is writable and does not alias `data`.
    return torch.from_numpy(array.astype(native)).reshape(shape)
