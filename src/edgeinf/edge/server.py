"""Edge gRPC server.

Phase 1 stub: Infer does no inference, it only echoes the size of the received tensor.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from concurrent import futures

import grpc

from edgeinf.common.grpc_options import DEFAULT_EDGE_PORT, GRPC_OPTIONS
from edgeinf.rpc import inference_pb2 as pb
from edgeinf.rpc import inference_pb2_grpc as pb_grpc

MAX_WORKERS = 8
SHUTDOWN_GRACE_S = 1.0
MS_PER_S = 1000.0


class InferenceServicer(pb_grpc.InferenceServicer):
    """Serves Infer and Health, and tracks how many requests are at the edge."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0

    def queue_len(self) -> int:
        """Number of Infer requests currently at the edge (waiting or in service)."""
        with self._lock:
            return self._in_flight

    def Infer(self, request: pb.InferRequest, context: grpc.ServicerContext) -> pb.InferReply:
        start = time.perf_counter()
        with self._lock:
            ahead = self._in_flight
            self._in_flight += 1
        try:
            return pb.InferReply(
                label=-1,
                confidence=0.0,
                queue_len=ahead,
                edge_ms=(time.perf_counter() - start) * MS_PER_S,
                tensor_bytes=len(request.tensor),
            )
        finally:
            with self._lock:
                self._in_flight -= 1

    def Health(self, request: pb.HealthRequest, context: grpc.ServicerContext) -> pb.HealthReply:
        return pb.HealthReply(ok=True, queue_len=self.queue_len())


def start_server(port: int = DEFAULT_EDGE_PORT, host: str = "[::]") -> tuple[grpc.Server, int]:
    """Start the server and return it with the bound port (pass port=0 to pick a free one)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS), options=GRPC_OPTIONS)
    pb_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server, bound_port


def main() -> None:
    server, port = start_server(int(os.environ.get("EDGE_PORT", DEFAULT_EDGE_PORT)))
    print(f"edge server listening on port {port}", flush=True)
    # As PID 1 in the container, SIGTERM is ignored unless handled, so `docker stop` would
    # hang until its kill timeout.
    signal.signal(signal.SIGTERM, lambda *_: server.stop(SHUTDOWN_GRACE_S))
    server.wait_for_termination()


if __name__ == "__main__":
    main()
