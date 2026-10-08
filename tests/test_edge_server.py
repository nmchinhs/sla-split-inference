"""The edge server (edge/server.py) through the device's RPC wrapper, over gRPC on localhost.

The models run on the CPU with random weights and settled BatchNorm statistics, so that their
outputs have an ordinary size (see conftest.py).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
import torch

from edgeinf.common.serialize import deserialize, serialize, serialize_jpeg
from edgeinf.data.imagenette import normalise
from edgeinf.device.client import EdgeClient
from edgeinf.edge.server import InferenceServicer, start_server
from edgeinf.models.multi_exit import INPUT_SIZE, MODEL_NAMES, build_model
from edgeinf.models.split import pack, run_head, top1
from edgeinf.profiler.layer_profile import raw_image

SEED = 0
PAYLOAD_BYTES = 5 * 1024 * 1024  # above gRPC's default 4 MiB message limit
STOP_GRACE_S = 0
SHORT_TIMEOUT_S = 0.5
WAIT_S = 5.0  # longest a test waits for another thread
POLL_S = 0.005
BLOCKED_MS = 50.0  # how long the test keeps the worker busy while others queue
CONFIDENCE_TOLERANCE = 0.01  # relative; 16-bit transfer moves the output a little
JPEG_QUALITY = 90
SMALL_SIDE = 64  # px, an image the models are not built for
MS_PER_S = 1e3
INVALID = grpc.StatusCode.INVALID_ARGUMENT


@pytest.fixture
def models(edge_models) -> dict:
    return edge_models


def _payload(model, frame, k: int) -> bytes:
    with torch.inference_mode():
        return pack(run_head(model, frame, k), k)


def _wait_until(condition) -> None:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for another thread"
        time.sleep(POLL_S)


def _block_worker(servicer, monkeypatch) -> tuple[threading.Event, list[int]]:
    """Make the edge's worker wait for the returned event before every job, noting each k."""
    release, served = threading.Event(), []
    answer = servicer._answer

    def slow(model, data, k):
        served.append(k)
        assert release.wait(WAIT_S)
        return answer(model, data, k)

    monkeypatch.setattr(servicer, "_answer", slow)
    return release, served


# ------------------------------------------------------------------------------ Health


def test_health_reports_ok_and_empty_queue(client):
    reply = client.health()
    assert reply is not None
    assert reply.ok
    assert reply.queue_len == 0
    assert reply.padding_bytes == 0


def test_health_probe_echoes_padding_size(client):
    reply = client.health(padding=bytes(PAYLOAD_BYTES))
    assert reply is not None
    assert reply.padding_bytes == PAYLOAD_BYTES


def test_unreachable_edge_fails_within_timeout():
    server, port = start_server(port=0, host="localhost")
    server.stop(STOP_GRACE_S)
    with EdgeClient(f"localhost:{port}", timeout_s=SHORT_TIMEOUT_S) as client:
        assert client.health() is None
        with pytest.raises(grpc.RpcError):
            client.infer(b"x", k=0)


# ------------------------------------------------------------------------------ Infer


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_infer_gives_the_whole_models_answer_at_every_split_point(models, client, frame, name):
    model = models[name]
    with torch.inference_mode():
        label, confidence = top1(model(normalise(frame)))
    for k in range(model.num_blocks):
        reply = client.infer(_payload(model, frame, k), k, request_id=k, model=name)
        assert reply.label == label, f"{name}: k={k}"
        assert reply.confidence == pytest.approx(confidence, rel=CONFIDENCE_TOLERANCE)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_infer_takes_the_frame_as_a_jpeg(models, client, frame, name):
    payload = serialize_jpeg(frame, JPEG_QUALITY)
    with torch.inference_mode():
        # JPEG is lossy: the edge answers for the image it decodes, not for the original.
        label, confidence = top1(models[name](normalise(deserialize(payload))))
    reply = client.infer(payload, 0, model=name)
    assert reply.label == label
    assert reply.confidence == pytest.approx(confidence, rel=CONFIDENCE_TOLERANCE)


def test_a_request_alone_does_not_wait_and_reports_its_compute_time(models, client, frame):
    reply = client.infer(_payload(models["resnet18"], frame, 0), 0, model="resnet18")
    assert reply.queue_len == 0
    assert 0 <= reply.queue_ms < BLOCKED_MS
    assert reply.compute_ms > 0


def _bad_requests(models, frame) -> dict[str, tuple[str, int, bytes]]:
    """Requests the edge must refuse, as (model name, k, tensor bytes)."""
    mobilenet, resnet = models["mobilenetv2"], models["resnet18"]
    n = mobilenet.num_blocks
    image = _payload(mobilenet, frame, 0)
    small = torch.zeros(1, 3, SMALL_SIDE, SMALL_SIDE, dtype=torch.uint8)
    with torch.inference_mode():
        after_block_4 = run_head(mobilenet, frame, 5)
    return {
        "unknown model": ("vgg16", 0, image),
        "no model named": ("", 0, image),
        "k = n": ("mobilenetv2", n, image),
        "k beyond n": ("mobilenetv2", n + 5, image),
        "k of the other model": ("resnet18", resnet.num_blocks + 2, image),
        "empty tensor": ("mobilenetv2", 0, b""),
        "bytes that are no tensor": ("mobilenetv2", 0, bytes(1000)),
        "truncated tensor": ("mobilenetv2", 0, image[:-1]),
        "tensor of another split point": ("mobilenetv2", 8, _payload(mobilenet, frame, 5)),
        "frame where a block output is due": ("mobilenetv2", 5, image),
        "block output where the frame is due": ("mobilenetv2", 0, _payload(mobilenet, frame, 5)),
        "float image at k = 0": ("mobilenetv2", 0, serialize(frame.float())),
        "8-bit block output": ("mobilenetv2", 5, serialize(after_block_4.to(torch.uint8))),
        "two frames": ("mobilenetv2", 0, serialize(frame.repeat(2, 1, 1, 1))),
        "frame of another size": ("mobilenetv2", 0, serialize(small)),
        "jpeg of another size": ("mobilenetv2", 0, serialize_jpeg(small, JPEG_QUALITY)),
        "jpeg where a block output is due": ("mobilenetv2", 5, serialize_jpeg(frame, JPEG_QUALITY)),
        "broken jpeg": ("mobilenetv2", 0, serialize_jpeg(frame, JPEG_QUALITY)[:1000]),
    }


BAD_REQUESTS = tuple(_bad_requests({name: build_model(name) for name in MODEL_NAMES}, raw_image(0)))


@pytest.mark.parametrize("case", BAD_REQUESTS)
def test_bad_requests_are_answered_with_invalid_argument(models, servicer, client, frame, case):
    name, k, data = _bad_requests(models, frame)[case]
    with pytest.raises(grpc.RpcError) as error:
        client.infer(data, k, model=name)
    assert error.value.code() == INVALID
    assert error.value.details()  # says what was wrong

    # The edge has forgotten the request and still serves.
    assert servicer.queue_len() == 0
    assert client.infer(_payload(models["resnet18"], frame, 0), 0, model="resnet18").queue_len == 0


def test_a_fault_while_computing_is_an_internal_error_and_the_edge_goes_on(
    models, servicer, client, frame, monkeypatch
):
    answer = servicer._answer
    faults = iter([RuntimeError("out of memory")])

    def faulty(model, data, k):
        for fault in faults:
            raise fault
        return answer(model, data, k)

    monkeypatch.setattr(servicer, "_answer", faulty)
    payload = _payload(models["resnet18"], frame, 0)
    with pytest.raises(grpc.RpcError) as error:
        client.infer(payload, 0, model="resnet18")
    # Not INVALID_ARGUMENT: the request was fine, the edge was not.
    assert error.value.code() == grpc.StatusCode.INTERNAL
    assert servicer.queue_len() == 0
    assert client.infer(payload, 0, model="resnet18").label >= 0


# ------------------------------------------------------------------------------ the queue


def test_requests_are_served_one_at_a_time_in_arrival_order(
    models, servicer, client, frame, monkeypatch
):
    model = models["resnet18"]
    release, served = _block_worker(servicer, monkeypatch)
    split_points = [3, 0, 6]  # told apart by k; sent in this order

    def timed_infer(payload: bytes, k: int, position: int):
        start = time.perf_counter()
        reply = client.infer(payload, k, position, "resnet18")
        return reply, (time.perf_counter() - start) * MS_PER_S

    with ThreadPoolExecutor(len(split_points)) as pool:
        futures = []
        for position, k in enumerate(split_points):
            futures.append(pool.submit(timed_infer, _payload(model, frame, k), k, position))
            _wait_until(lambda: servicer.queue_len() == position + 1)  # noqa: B023

        # The first is in service, the others wait: Health sees all three and is not held up.
        assert served == split_points[:1]
        health = client.health()
        assert health is not None and health.queue_len == len(split_points)

        time.sleep(BLOCKED_MS / MS_PER_S)
        release.set()
        replies, round_trips = zip(*(future.result(WAIT_S) for future in futures), strict=True)

    assert served == split_points
    # queue_len is what a request met on arrival, not counting itself.
    assert [reply.queue_len for reply in replies] == [0, 1, 2]
    # The first started at once; the others waited at least as long as the worker was held.
    # (The third need not have waited longer than the second: it also arrived later.)
    assert replies[0].queue_ms < BLOCKED_MS
    assert replies[1].queue_ms >= BLOCKED_MS and replies[2].queue_ms >= BLOCKED_MS
    # The first was held in service, so the wait shows up as its compute time.
    assert replies[0].compute_ms >= BLOCKED_MS
    # Waiting and computing do not overlap: together they fit into the device's round trip.
    for reply, round_trip_ms in zip(replies, round_trips, strict=True):
        assert reply.queue_ms + reply.compute_ms <= round_trip_ms
    assert servicer.queue_len() == 0
    assert client.health().queue_len == 0


def test_infer_gives_up_at_the_devices_timeout(models, servicer, client, frame, monkeypatch):
    release, _ = _block_worker(servicer, monkeypatch)
    start = time.monotonic()
    with pytest.raises(grpc.RpcError) as error:
        client.infer(
            _payload(models["resnet18"], frame, 0), 0, model="resnet18", timeout_s=SHORT_TIMEOUT_S
        )
    elapsed_s = time.monotonic() - start
    release.set()
    assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    assert SHORT_TIMEOUT_S <= elapsed_s < 2 * SHORT_TIMEOUT_S
    _wait_until(lambda: servicer.queue_len() == 0)  # the edge finishes the abandoned request


# ------------------------------------------------------------------------------ setup


def test_edge_computes_without_tf32(models):
    # TF32 would make the GPU edge differ from the CPU devices (CLAUDE.md, edge numerics).
    before = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    try:
        InferenceServicer(models.values(), device="cpu").close()
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before


def test_edge_knows_what_every_block_takes(models, servicer):
    for name, model in models.items():
        inputs = servicer.block_inputs(name)
        assert len(inputs) == model.num_blocks
        assert inputs[0] == (torch.uint8, (1, 3, INPUT_SIZE, INPUT_SIZE))  # the raw frame
        assert all(dtype == torch.float32 for dtype, _ in inputs[1:])
        with torch.inference_mode():
            x = normalise(raw_image(SEED))
            for k in range(1, model.num_blocks):
                x = model.blocks[k - 1](x)
                assert inputs[k][1] == tuple(x.shape)


def test_a_server_without_models_answers_health_and_refuses_infer(frame):
    server, port = start_server(port=0, host="localhost")
    try:
        with EdgeClient(f"localhost:{port}") as client:
            assert client.health().ok
            with pytest.raises(grpc.RpcError) as error:
                client.infer(serialize(frame), 0, model="mobilenetv2")
            assert error.value.code() == INVALID
    finally:
        server.stop(STOP_GRACE_S)
