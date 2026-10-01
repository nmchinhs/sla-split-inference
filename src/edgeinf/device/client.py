"""Device-side gRPC client for the edge.

Phase 1 stub: it only wraps the two RPCs with timeouts. Exit checks, scheduling and local
fallback are added later. Must stay free of CUDA calls.
"""

from __future__ import annotations

import grpc

from edgeinf.common.grpc_options import GRPC_OPTIONS
from edgeinf.rpc import inference_pb2 as pb
from edgeinf.rpc import inference_pb2_grpc as pb_grpc

DEFAULT_TIMEOUT_S = 2.0


class EdgeClient:
    """Talks to the edge at `addr` ("host:port"). Every call has a timeout in seconds."""

    def __init__(self, addr: str, device_id: str = "", timeout_s: float = DEFAULT_TIMEOUT_S):
        self.device_id = device_id
        self.timeout_s = timeout_s
        self._channel = grpc.insecure_channel(addr, options=GRPC_OPTIONS)
        self._stub = pb_grpc.InferenceStub(self._channel)

    def health(self, timeout_s: float | None = None) -> pb.HealthReply | None:
        """Return the edge's health reply, or None if it is unreachable or timed out."""
        try:
            return self._stub.Health(pb.HealthRequest(), timeout=timeout_s or self.timeout_s)
        except grpc.RpcError:
            return None

    def infer(
        self,
        tensor: bytes,
        k: int,
        request_id: int = 0,
        model: str = "",
        timeout_s: float | None = None,
    ) -> pb.InferReply:
        """Send `tensor` for blocks [k, n). Raises grpc.RpcError on timeout or failure."""
        request = pb.InferRequest(
            tensor=tensor,
            k=k,
            meta=pb.Meta(device_id=self.device_id, request_id=request_id, model=model),
        )
        return self._stub.Infer(request, timeout=timeout_s or self.timeout_s)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> EdgeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
