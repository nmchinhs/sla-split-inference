from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Meta(_message.Message):
    __slots__ = ("device_id", "request_id", "model")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    request_id: int
    model: str
    def __init__(self, device_id: _Optional[str] = ..., request_id: _Optional[int] = ..., model: _Optional[str] = ...) -> None: ...

class InferRequest(_message.Message):
    __slots__ = ("tensor", "k", "meta")
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    tensor: bytes
    k: int
    meta: Meta
    def __init__(self, tensor: _Optional[bytes] = ..., k: _Optional[int] = ..., meta: _Optional[_Union[Meta, _Mapping]] = ...) -> None: ...

class InferReply(_message.Message):
    __slots__ = ("label", "confidence", "queue_len", "queue_ms", "compute_ms")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_LEN_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_MS_FIELD_NUMBER: _ClassVar[int]
    label: int
    confidence: float
    queue_len: int
    queue_ms: float
    compute_ms: float
    def __init__(self, label: _Optional[int] = ..., confidence: _Optional[float] = ..., queue_len: _Optional[int] = ..., queue_ms: _Optional[float] = ..., compute_ms: _Optional[float] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ("padding",)
    PADDING_FIELD_NUMBER: _ClassVar[int]
    padding: bytes
    def __init__(self, padding: _Optional[bytes] = ...) -> None: ...

class HealthReply(_message.Message):
    __slots__ = ("ok", "queue_len", "padding_bytes")
    OK_FIELD_NUMBER: _ClassVar[int]
    QUEUE_LEN_FIELD_NUMBER: _ClassVar[int]
    PADDING_BYTES_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    queue_len: int
    padding_bytes: int
    def __init__(self, ok: _Optional[bool] = ..., queue_len: _Optional[int] = ..., padding_bytes: _Optional[int] = ...) -> None: ...
