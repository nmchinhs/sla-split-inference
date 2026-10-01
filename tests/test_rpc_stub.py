"""The Phase 1 stub edge server and device client talk over gRPC on localhost."""

import grpc
import pytest

from edgeinf.device.client import EdgeClient
from edgeinf.edge.server import start_server

PAYLOAD_BYTES = 5 * 1024 * 1024  # above gRPC's default 4 MiB message limit
STOP_GRACE_S = 0
SHORT_TIMEOUT_S = 0.5


@pytest.fixture
def edge_addr():
    server, port = start_server(port=0, host="localhost")
    yield f"localhost:{port}"
    server.stop(STOP_GRACE_S)


def test_health_reports_ok_and_empty_queue(edge_addr):
    with EdgeClient(edge_addr) as client:
        reply = client.health()
    assert reply is not None
    assert reply.ok
    assert reply.queue_len == 0


def test_infer_echoes_tensor_size(edge_addr):
    with EdgeClient(edge_addr, device_id="device-1") as client:
        reply = client.infer(bytes(PAYLOAD_BYTES), k=3, request_id=7)
    assert reply.tensor_bytes == PAYLOAD_BYTES
    assert reply.queue_len == 0


def test_unreachable_edge_fails_within_timeout():
    server, port = start_server(port=0, host="localhost")
    server.stop(STOP_GRACE_S)
    with EdgeClient(f"localhost:{port}", timeout_s=SHORT_TIMEOUT_S) as client:
        assert client.health() is None
        with pytest.raises(grpc.RpcError):
            client.infer(b"x", k=0)
