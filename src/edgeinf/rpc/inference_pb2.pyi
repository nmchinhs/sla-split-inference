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
    def __init__(
        self,
        device_id: _Optional[str] = ...,
        request_id: _Optional[int] = ...,
        model: _Optional[str] = ...,
    ) -> None: ...

class InferRequest(_message.Message):
    __slots__ = ("tensor", "k", "meta")
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    tensor: bytes
    k: int
    meta: Meta
    def __init__(
        self,
        tensor: _Optional[bytes] = ...,
        k: _Optional[int] = ...,
        meta: _Optional[_Union[Meta, _Mapping]] = ...,
    ) -> None: ...

class InferReply(_message.Message):
    __slots__ = ("label", "confidence", "queue_len", "edge_ms", "tensor_bytes")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_LEN_FIELD_NUMBER: _ClassVar[int]
    EDGE_MS_FIELD_NUMBER: _ClassVar[int]
    TENSOR_BYTES_FIELD_NUMBER: _ClassVar[int]
    label: int
    confidence: float
    queue_len: int
    edge_ms: float
    tensor_bytes: int
    def __init__(
        self,
        label: _Optional[int] = ...,
        confidence: _Optional[float] = ...,
        queue_len: _Optional[int] = ...,
        edge_ms: _Optional[float] = ...,
        tensor_bytes: _Optional[int] = ...,
    ) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthReply(_message.Message):
    __slots__ = ("ok", "queue_len")
    OK_FIELD_NUMBER: _ClassVar[int]
    QUEUE_LEN_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    queue_len: int
    def __init__(self, ok: _Optional[bool] = ..., queue_len: _Optional[int] = ...) -> None: ...
