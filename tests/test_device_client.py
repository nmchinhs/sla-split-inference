"""The device runtime (device/client.py): plans, one request with fallback, and arrivals.

Requests go to a real edge server on localhost (conftest.py). The arrival loop is tested with a
clock that only moves when the test says so, which makes every latency exact.
"""

import io
import json
import logging
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from edgeinf.common.logging import FIELDS
from edgeinf.common.serialize import deserialize, serialize_jpeg
from edgeinf.data.imagenette import FRAMES_FILE, LABELS_FILE, normalise
from edgeinf.device import client as device_client
from edgeinf.device.bandwidth import PROBE_INTERVAL_S, PROBE_START_BYTES, LinkEstimator
from edgeinf.device.client import Agent, Answer, Device, EdgeClient, Plan, fixed_plan, serve
from edgeinf.edge.server import start_server
from edgeinf.models import split
from edgeinf.models.multi_exit import add_exits, build_model
from edgeinf.models.split import exit_check, pack, run_blocks, run_head, run_tail, top1
from edgeinf.profiler.layer_profile import raw_image
from edgeinf.rpc import inference_pb2 as pb

RAW_IMAGE_BYTES = 150_550  # 3 x 224 x 224 bytes and a 22-byte header
JPEG_QUALITY = 90
SPLIT_POINTS = (0, 5)  # everything offloaded, and a real split
TIMEOUT_S = 0.3
WAIT_S = 5.0
MS_PER_S = 1e3
STOP_GRACE_S = 0
WALL_START = 1_791_450_000.0  # an arbitrary Unix time for the fake wall clock
N_FRAMES = 6
FRAME_SEEDS = range(10, 10 + N_FRAMES)
EXITS = {"mobilenetv2": [4, 10, 14], "resnet18": [2, 4, 6]}
HEAD_POOL, HEAD_HIDDEN = 7, 16
EXIT_MODEL_SEED = 3
TAU_ALWAYS = 1e-6  # every confidence is at least this
TAU_ALMOST_NEVER = 1.0 - 1e-6  # no head of random weights is this sure


@pytest.fixture(params=["mobilenetv2", "resnet18"])
def model(request, edge_models):
    return edge_models[request.param]


@pytest.fixture
def expected(model, frame) -> int:
    """The label of the whole model, which every way of serving the frame must give."""
    with torch.inference_mode():
        return top1(model(normalise(frame)))[0]


@pytest.fixture(autouse=True)
def _inference_mode():
    with torch.inference_mode():
        yield


def _block_worker(servicer, monkeypatch) -> threading.Event:
    """Make the edge's worker wait for the returned event before every job."""
    release = threading.Event()
    answer = servicer._answer

    def slow(model, data, k):
        assert release.wait(WAIT_S)
        return answer(model, data, k)

    monkeypatch.setattr(servicer, "_answer", slow)
    return release


# ------------------------------------------------------------------------------ plans


def test_local_runs_every_block_on_the_device():
    assert fixed_plan("local", 20) == Plan("local", k=20, tau=1.0, jpeg_quality=None)


def test_offload_sends_the_raw_frame():
    assert fixed_plan("offload", 20) == Plan("offload", k=0, tau=1.0, jpeg_quality=None)
    # A quality in the config belongs to offload_jpeg and must not turn offload into it.
    assert fixed_plan("offload", 20, jpeg_quality=75).jpeg_quality is None


def test_offload_jpeg_sends_the_frame_as_a_jpeg_of_the_configured_quality():
    assert fixed_plan("offload_jpeg", 20, jpeg_quality=75) == Plan(
        "offload_jpeg", k=0, tau=1.0, jpeg_quality=75
    )
    with pytest.raises(ValueError, match="jpeg_quality"):
        fixed_plan("offload_jpeg", 20)


@pytest.mark.parametrize("variant", ["joint", "split_only", "exit_only", "pinned", "cloud"])
def test_variants_that_need_a_scheduler_are_not_fixed_plans(variant):
    with pytest.raises(ValueError, match=variant):
        fixed_plan(variant, 20)


# ------------------------------------------------------------------------------ one request


def test_local_answers_on_the_device_and_sends_nothing(model, frame, expected):
    device = Device(model, client=None)
    answer = device.infer(frame, fixed_plan("local", model.num_blocks))
    assert answer.label == expected
    assert answer.bytes_sent == 0
    assert answer.rpc_ms == 0
    assert not answer.exited_on_device  # the last block answered, not an exit head
    assert (answer.edge_queue_len, answer.edge_queue_ms, answer.edge_compute_ms) == (None,) * 3


def test_offload_sends_the_raw_frame_and_takes_the_edges_answer(model, frame, expected, client):
    answer = Device(model, client).infer(frame, fixed_plan("offload", model.num_blocks))
    assert answer.label == expected
    assert answer.bytes_sent == RAW_IMAGE_BYTES
    assert answer.edge_up
    assert answer.edge_queue_len == 0
    assert answer.edge_compute_ms > 0 and answer.edge_queue_ms >= 0
    # The round trip contains the edge's own time.
    assert answer.rpc_ms > answer.edge_queue_ms + answer.edge_compute_ms


def test_offload_jpeg_sends_fewer_bytes_and_answers_for_the_decoded_image(model, frame, client):
    plan = fixed_plan("offload_jpeg", model.num_blocks, JPEG_QUALITY)
    answer = Device(model, client).infer(frame, plan)
    payload = serialize_jpeg(frame, JPEG_QUALITY)
    assert answer.bytes_sent == len(payload) < RAW_IMAGE_BYTES
    assert answer.label == top1(model(normalise(deserialize(payload))))[0]
    assert answer.edge_up


def test_a_split_plan_sends_the_block_output(model, frame, expected, client):
    k = 5
    answer = Device(model, client).infer(frame, Plan("split_only", k))
    assert answer.label == expected
    assert answer.bytes_sent == len(pack(run_head(model, frame, k), k))


def test_the_edge_is_told_which_request_it_serves(model, frame, servicer, client, monkeypatch):
    seen = []
    infer = client.infer

    def spy(tensor, k, request_id=0, model="", timeout_s=None):
        seen.append((request_id, model, k))
        return infer(tensor, k, request_id, model, timeout_s)

    monkeypatch.setattr(client, "infer", spy)
    device = Device(model, client)
    for _ in range(3):
        device.infer(frame, fixed_plan("offload", model.num_blocks))
    assert seen == [(0, model.name, 0), (1, model.name, 0), (2, model.name, 0)]


# ------------------------------------------------------------------------------ fallback


@pytest.mark.parametrize("k", SPLIT_POINTS)
def test_unreachable_edge_means_the_device_finishes_alone(model, frame, expected, caplog, k):
    server, port = start_server(port=0, host="localhost")
    server.stop(STOP_GRACE_S)
    with EdgeClient(f"localhost:{port}", timeout_s=TIMEOUT_S) as client:
        device = Device(model, client)
        with caplog.at_level(logging.WARNING):
            first = device.infer(frame, Plan("split_only", k))
            second = device.infer(frame, Plan("split_only", k))

    assert first.label == second.label == expected
    assert not first.edge_up and not second.edge_up
    assert not device.edge_up
    # The first request was sent and lost. The second was not sent: the probe said the edge is
    # still away, so the device did not wait for a timeout again.
    assert first.bytes_sent == len(pack(run_head(model, frame, k), k))
    assert second.bytes_sent == 0
    assert (first.edge_queue_len, first.edge_compute_ms) == (None, None)
    fallbacks = [record for record in caplog.records if "fallback" in record.getMessage()]
    assert len(fallbacks) == 2  # every fallback is logged
    assert "UNAVAILABLE" in fallbacks[0].getMessage()


def test_a_timeout_means_the_device_finishes_alone_and_the_wait_is_counted(
    model, frame, expected, servicer, edge_addr, monkeypatch, caplog
):
    release = _block_worker(servicer, monkeypatch)
    with EdgeClient(edge_addr, timeout_s=TIMEOUT_S) as client:
        device = Device(model, client)
        with caplog.at_level(logging.WARNING):
            answer = device.infer(frame, fixed_plan("offload", model.num_blocks))
        release.set()

    assert answer.label == expected
    assert not answer.edge_up
    assert answer.bytes_sent == RAW_IMAGE_BYTES
    assert answer.rpc_ms >= TIMEOUT_S * MS_PER_S
    assert "DEADLINE_EXCEEDED" in caplog.text and "fallback" in caplog.text


def test_the_device_uses_the_edge_again_once_it_answers(
    model, frame, expected, servicer, edge_addr, monkeypatch
):
    release = _block_worker(servicer, monkeypatch)
    plan = fixed_plan("offload", model.num_blocks)
    with EdgeClient(edge_addr, timeout_s=TIMEOUT_S) as client:
        device = Device(model, client)
        assert not device.infer(frame, plan).edge_up  # times out
        release.set()
        client.timeout_s = WAIT_S
        answer = device.infer(frame, plan)

    assert answer.edge_up and device.edge_up
    assert answer.label == expected
    assert answer.edge_queue_len is not None  # answered by the edge, not locally


class _UnhealthyEdge:
    """An edge that is reachable but says it is not ready. It must not be sent any request."""

    timeout_s = TIMEOUT_S

    def __init__(self) -> None:
        self.probes = 0

    def health(self, padding: bytes = b"", timeout_s: float | None = None) -> pb.HealthReply:
        self.probes += 1
        return pb.HealthReply(ok=False)

    def infer(self, *args, **kwargs):
        raise AssertionError("a request was sent to an edge known to be unhealthy")


def test_an_unhealthy_edge_is_probed_and_not_used(model, frame, expected, caplog):
    edge = _UnhealthyEdge()
    device = Device(model, edge)
    device.edge_up = False  # as after a failed request
    with caplog.at_level(logging.WARNING):
        answer = device.infer(frame, fixed_plan("offload", model.num_blocks))
    assert edge.probes == 1
    assert answer.label == expected
    assert not answer.edge_up and answer.bytes_sent == 0
    assert "fallback" in caplog.text


def test_a_local_request_does_not_touch_the_edge(model, frame, expected):
    edge = _UnhealthyEdge()
    device = Device(model, edge)
    device.edge_up = False
    assert device.infer(frame, fixed_plan("local", model.num_blocks)).label == expected
    assert edge.probes == 0


def test_a_device_without_an_edge_serves_every_plan_locally(model, frame, expected, caplog):
    device = Device(model, client=None)
    with caplog.at_level(logging.WARNING):
        answer = device.infer(frame, fixed_plan("offload", model.num_blocks))
    assert answer.label == expected and not answer.edge_up
    assert "fallback" in caplog.text


# ------------------------------------------------------------------------------ exits


@pytest.fixture(scope="module", params=["mobilenetv2", "resnet18"])
def exit_model(request, settle_batchnorm):
    """A model of ordinary scale with three exit heads of random weights."""
    torch.manual_seed(EXIT_MODEL_SEED)
    model = build_model(request.param)
    add_exits(model, EXITS[request.param], HEAD_POOL, HEAD_HIDDEN)
    settle_batchnorm(model)
    return model.eval()


def _first_exit(model, frame) -> tuple[int, int]:
    """(block, label) of the model's first exit head for `frame`."""
    block = model.exit_points[0]
    return block, exit_check(model, run_blocks(model, normalise(frame), 0, block + 1), block)[0]


def test_an_exit_that_is_sure_enough_answers_and_nothing_is_sent(exit_model, frame, client, caplog):
    block, label = _first_exit(exit_model, frame)
    device = Device(exit_model, client)
    with caplog.at_level(logging.WARNING):
        answer = device.infer(frame, Plan("pinned", k=block + 2, tau=TAU_ALWAYS))
    assert answer.exited_on_device
    assert answer.label == label
    assert (answer.bytes_sent, answer.rpc_ms) == (0, 0.0)
    assert answer.edge_queue_len is None  # the edge was not asked
    assert not caplog.records  # an exit is not a fallback


def test_a_frame_no_exit_is_sure_of_goes_on_to_the_edge(exit_model, frame, client, edge_models):
    k = exit_model.exit_points[1] + 1  # two exits are checked on the way
    answer = Device(exit_model, client).infer(frame, Plan("pinned", k=k, tau=TAU_ALMOST_NEVER))
    assert not answer.exited_on_device
    sent = pack(run_head(exit_model, frame, k), k)
    assert answer.bytes_sent == len(sent)
    # The test's edge holds other weights than the device: the answer is the edge's.
    edge_model = edge_models[exit_model.name]
    assert answer.label == top1(run_tail(edge_model, deserialize(sent), k))[0]
    assert answer.edge_up


def test_exits_also_answer_when_everything_runs_on_the_device(exit_model, frame):
    n = exit_model.num_blocks
    device = Device(exit_model, client=None)
    assert device.infer(frame, Plan("pinned", k=n, tau=TAU_ALWAYS)).exited_on_device
    answer = device.infer(frame, Plan("pinned", k=n, tau=TAU_ALMOST_NEVER))
    assert not answer.exited_on_device
    assert answer.label == top1(exit_model(normalise(frame)))[0]


def test_an_exit_behind_the_split_point_is_not_checked(exit_model, frame, client):
    k = exit_model.exit_points[0]  # the head reads block k, which the device does not run
    answer = Device(exit_model, client).infer(frame, Plan("pinned", k=k, tau=TAU_ALWAYS))
    assert not answer.exited_on_device and answer.bytes_sent > 0


def test_with_tau_one_no_head_is_run(exit_model, frame, monkeypatch):
    def no_head(*_args):
        raise AssertionError("an exit head was run")

    monkeypatch.setattr(split, "exit_check", no_head)
    n = exit_model.num_blocks
    answer = Device(exit_model, client=None).infer(frame, fixed_plan("local", n))
    assert not answer.exited_on_device


# ------------------------------------------------------------------------------ the link


def _recorded_health(client, monkeypatch) -> list[int]:
    """Make `client` note the padding size of every Health call it makes."""
    sizes: list[int] = []
    health = client.health

    def noting(padding: bytes = b"", timeout_s: float | None = None):
        sizes.append(len(padding))
        return health(padding, timeout_s)

    monkeypatch.setattr(client, "health", noting)
    return sizes


def test_an_answered_infer_tells_the_device_about_the_edge_queue(model, frame, client):
    device = Device(model, client)
    device.link.add_rtt(0.0)
    device.infer(frame, fixed_plan("offload", model.num_blocks))
    assert device.link.queue_len == 0
    assert device.link.bw_mbps is not None  # the transfer was a sample, or at least a floor


def test_a_probe_measures_the_rtt_and_then_the_bandwidth(model, client, monkeypatch):
    sizes = _recorded_health(client, monkeypatch)
    device = Device(model, client)
    device.probe(budget_s=WAIT_S)
    assert sizes == [0, PROBE_START_BYTES]  # an empty call, then one with padding
    assert device.link.rtt_ms > 0
    assert device.link.bw_mbps > 0
    assert device.link.queue_len == 0


def test_no_probe_is_made_while_the_estimates_are_fresh(model, client, monkeypatch):
    now = [0.0]
    device = Device(model, client, LinkEstimator(clock=lambda: now[0]))
    device.link.add_rtt(1.0)
    device.link.add_transfer(100_000, 101.0)
    sizes = _recorded_health(client, monkeypatch)
    device.probe(budget_s=WAIT_S)
    assert sizes == []
    now[0] += PROBE_INTERVAL_S + 1
    device.probe(budget_s=WAIT_S)
    assert len(sizes) == 2


def test_a_probe_that_does_not_fit_before_the_next_request_is_not_made(model, client, monkeypatch):
    sizes = _recorded_health(client, monkeypatch)
    device = Device(model, client)
    device.probe(budget_s=0.0)
    assert sizes == []
    # Room for the RTT probe only: the bandwidth probe waits for a longer pause.
    device.link.add_rtt(100.0)
    device.link.add_transfer(100_000, 8_100.0)  # 0.1 Mbit/s: even the smallest probe is slow
    device.link = _aged(device.link)
    device.probe(budget_s=0.16)
    assert sizes == [0]


def _aged(link: LinkEstimator) -> LinkEstimator:
    """`link` with both estimates due for a probe."""
    link._rtt_at = link._bw_at = float("-inf")
    return link


def test_a_device_without_an_edge_or_with_an_edge_that_is_down_does_not_probe(
    model, client, monkeypatch
):
    Device(model, client=None).probe(budget_s=WAIT_S)  # nothing to ask: must not raise
    sizes = _recorded_health(client, monkeypatch)
    device = Device(model, client)
    device.edge_up = False
    device.probe(budget_s=WAIT_S)
    assert sizes == []


# ------------------------------------------------------------------------------ arrivals


class _Clock:
    """Time that moves only when someone sleeps or a request is served."""

    def __init__(self) -> None:
        self.now = 100.0  # perf_counter-like: an arbitrary origin
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def wall(self) -> float:
        return WALL_START + (self.now - 100.0)


class _ScriptedDevice:
    """Stands in for Device: each request takes the next scripted time and gets the next label."""

    def __init__(self, clock: _Clock, service_s: list[float], labels: list[int]) -> None:
        self.clock, self.service_s, self.labels = clock, list(service_s), list(labels)
        self.started_at: list[float] = []
        self.frames: list[int] = []
        self.link = LinkEstimator(clock=clock)
        self.probes: list[tuple[float, float]] = []  # (when, budget in s)
        self.bw_after: list[float] = []  # the bandwidth estimate after each request

    def probe(self, budget_s: float) -> None:
        self.probes.append((self.clock.now, budget_s))

    def infer(self, frame: torch.Tensor, plan: Plan) -> Answer:
        self.started_at.append(self.clock.now)
        self.frames.append(int(frame[0, 0, 0, 0]))
        self.clock.now += self.service_s.pop(0)
        if self.bw_after:
            self.link.bw_mbps = self.bw_after.pop(0)
        return Answer(
            label=self.labels.pop(0),
            exited_on_device=False,
            bytes_sent=1000,
            edge_up=True,
            rpc_ms=20.0,
            edge_queue_len=1,
            edge_queue_ms=2.0,
            edge_compute_ms=3.0,
        )


def _numbered_frames(count: int) -> torch.Tensor:
    """Frames whose first pixel is their index, shape (count, 3, 2, 2)."""
    return torch.arange(count, dtype=torch.uint8).view(count, 1, 1, 1).expand(count, 3, 2, 2)


def _serve(clock, device, labels, rate_hz, warmup=0, plan=None):
    return serve(
        device,
        _numbered_frames(len(labels)),
        labels,
        plan or Plan("offload", 0),
        rate_hz=rate_hz,
        warmup=warmup,
        device_id="device-7",
        clock=clock,
        sleep=clock.sleep,
        wall=clock.wall,
    )


def test_latency_runs_from_the_scheduled_arrival_and_includes_the_wait():
    # 10 requests per second: due at 0, 100, 200, 300 ms. The second takes 250 ms, so the third
    # and fourth find the device busy and wait.
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.05, 0.25, 0.05, 0.05], labels=[0, 1, 2, 3])
    rows = _serve(clock, device, labels=[0, 1, 2, 3], rate_hz=10)

    assert len(rows) == 4  # nothing is dropped
    assert device.frames == [0, 1, 2, 3]  # served in arrival order
    assert [row["latency_ms"] for row in rows] == pytest.approx([50, 250, 200, 150])
    assert [row["wait_ms"] for row in rows] == pytest.approx([0, 0, 150, 100], abs=1e-6)
    # Starts: at once, at its due time, then as soon as the device is free.
    assert device.started_at == pytest.approx([100.0, 100.1, 100.35, 100.4])
    assert [row["timestamp"] - WALL_START for row in rows] == pytest.approx([0, 0.1, 0.2, 0.3])


def test_an_idle_device_waits_for_the_next_arrival():
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.03, 0.03, 0.03], labels=[0, 0, 0])
    rows = _serve(clock, device, labels=[0, 0, 0], rate_hz=5)
    assert clock.slept == pytest.approx([0.17, 0.17])  # 200 ms apart, 30 ms of work
    assert [row["wait_ms"] for row in rows] == pytest.approx([0, 0, 0], abs=1e-6)
    assert [row["latency_ms"] for row in rows] == pytest.approx([30, 30, 30])


def test_warmup_requests_are_served_but_not_logged():
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.05] * 5, labels=[0] * 5)
    rows = _serve(clock, device, labels=[0] * 5, rate_hz=10, warmup=2)
    assert device.frames == [0, 1, 2, 3, 4]
    assert len(rows) == 3
    # The arrival clock does not restart after the warm-up: the third request is due at 200 ms.
    assert rows[0]["timestamp"] - WALL_START == pytest.approx(0.2)


def test_back_to_back_starts_each_request_when_the_last_one_ends():
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.1, 0.3, 0.2], labels=[0, 0, 0])
    rows = _serve(clock, device, labels=[0, 0, 0], rate_hz=None)
    assert clock.slept == []
    assert [row["latency_ms"] for row in rows] == pytest.approx([100, 300, 200])
    assert [row["wait_ms"] for row in rows] == pytest.approx([0, 0, 0], abs=1e-6)
    assert [row["timestamp"] - WALL_START for row in rows] == pytest.approx([0, 0.1, 0.4])


@pytest.mark.parametrize("rate_hz", [0, -5])
def test_a_rate_that_is_not_positive_is_refused(rate_hz):
    clock = _Clock()
    with pytest.raises(ValueError, match="rate"):
        _serve(clock, _ScriptedDevice(clock, [0.1], [0]), labels=[0], rate_hz=rate_hz)


def test_a_row_has_the_log_columns_and_says_what_happened():
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.05, 0.05], labels=[4, 9])
    plan = Plan("offload_jpeg", k=0, tau=1.0, jpeg_quality=75)
    first, second = _serve(clock, device, labels=[4, 7], rate_hz=10, plan=plan)

    assert tuple(first) == FIELDS
    assert (first["device_id"], first["variant"], first["k"], first["tau"]) == (
        "device-7",
        "offload_jpeg",
        0,
        1.0,
    )
    assert first["correct"] is True and second["correct"] is False  # label 9 is not 7
    assert first["exited_on_device"] is False and first["edge_up"] is True
    assert (first["bytes_sent"], first["edge_queue_len"]) == (1000, 1)
    assert first["bw_est_mbps"] is None  # nothing has been measured yet
    assert (first["rpc_ms"], first["edge_queue_ms"], first["edge_compute_ms"]) == (20.0, 2.0, 3.0)
    assert first["device_ms"] == pytest.approx(30.0)  # 50 ms of service, 20 of them waiting
    for row in (first, second):
        assert row["latency_ms"] == pytest.approx(row["wait_ms"] + row["device_ms"] + row["rpc_ms"])
    json.dumps(first)  # the agent sends rows to the host as JSON


def test_a_row_carries_the_split_point_and_threshold_of_its_plan():
    clock = _Clock()
    plan = Plan("joint", k=7, tau=0.8)
    (row,) = _serve(clock, _ScriptedDevice(clock, [0.05], [0]), labels=[0], rate_hz=10, plan=plan)
    assert (row["variant"], row["k"], row["tau"]) == ("joint", 7, 0.8)


def test_a_row_carries_the_bandwidth_estimate_the_request_started_with():
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.05, 0.05, 0.05], labels=[0, 0, 0])
    device.bw_after = [4.0, 5.0, 6.0]
    rows = _serve(clock, device, labels=[0, 0, 0], rate_hz=10)
    assert [row["bw_est_mbps"] for row in rows] == [None, 4.0, 5.0]


def test_the_device_may_probe_in_the_pause_before_the_next_arrival():
    # 10 requests per second. The first takes 30 ms and leaves 70 ms, the second takes 150 ms
    # and leaves none, the third is the last: nothing arrives after it.
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.03, 0.15, 0.03], labels=[0, 0, 0])
    _serve(clock, device, labels=[0, 0, 0], rate_hz=10, warmup=1)
    assert len(device.probes) == 1
    when, budget_s = device.probes[0]
    assert when == pytest.approx(100.03)
    assert budget_s == pytest.approx(0.07)


def test_back_to_back_requests_leave_no_pause_to_probe_in():
    clock = _Clock()
    device = _ScriptedDevice(clock, [0.03, 0.03], labels=[0, 0])
    _serve(clock, device, labels=[0, 0], rate_hz=None)
    assert device.probes == []


# ------------------------------------------------------------------------------ the agent


@pytest.fixture
def workspace(model, tmp_path):
    """A checkpoint of `model` and a small frames file, as a device container has them."""
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.name, "state_dict": model.state_dict()}, checkpoint)
    frames = torch.cat([raw_image(seed) for seed in FRAME_SEEDS])
    labels = [top1(model(normalise(frame[None])))[0] for frame in frames]
    labels[1] = (labels[1] + 1) % 10  # one frame that the model gets wrong
    np.save(tmp_path / FRAMES_FILE.format(split="test"), frames.numpy())
    np.save(tmp_path / LABELS_FILE.format(split="test"), np.asarray(labels))
    return {
        "checkpoint": str(checkpoint),
        "frames_dir": str(tmp_path),
        "split": "test",
        "rate_hz": 200.0,
        "timeout_s": WAIT_S,
        "jpeg_quality": JPEG_QUALITY,
    }


def test_a_job_serves_the_chosen_frames_and_returns_one_row_each(workspace, edge_addr, model):
    agent = Agent("device-3", edge_addr)
    job = {**workspace, "variant": "offload", "indices": [4, 1, 0, 5], "warmup": 1}
    reply = agent.run(job)

    rows = reply["rows"]
    assert len(rows) == 3  # four requests, the first a warm-up
    assert all(tuple(row) == FIELDS for row in rows)
    assert [row["correct"] for row in rows] == [False, True, True]  # frame 1 is the wrong one
    assert {row["device_id"] for row in rows} == {"device-3"}
    assert {row["bytes_sent"] for row in rows} == {RAW_IMAGE_BYTES}
    assert all(row["edge_up"] for row in rows)
    assert isinstance(reply["minor_faults"], int)
    assert reply["n_blocks"] == model.num_blocks  # tells a reader of the CSV which k is "local"
    json.dumps(reply)


def test_a_job_serves_exactly_the_frames_it_names_in_their_order(workspace, edge_addr):
    # Every frame gives a JPEG of another size, so the bytes sent tell which frame was served.
    frames = np.load(Path(workspace["frames_dir"]) / FRAMES_FILE.format(split="test"))
    sizes = [len(serialize_jpeg(torch.from_numpy(f[None]), JPEG_QUALITY)) for f in frames]
    assert len(set(sizes)) == len(sizes)

    indices = [4, 1, 0, 5]
    job = {**workspace, "variant": "offload_jpeg", "indices": indices, "warmup": 0}
    rows = Agent("device-3", edge_addr).run(job)["rows"]
    assert [row["bytes_sent"] for row in rows] == [sizes[index] for index in indices]


def test_a_job_can_run_any_fixed_variant_and_keeps_the_model_loaded(workspace, edge_addr):
    agent = Agent("device-3", edge_addr)
    sent = {}
    for variant in ("local", "offload", "offload_jpeg"):
        job = {**workspace, "variant": variant, "indices": [0, 2, 3], "warmup": 0}
        rows = agent.run(job)["rows"]
        assert [row["correct"] for row in rows] == [True, True, True] or variant == "offload_jpeg"
        sent[variant] = [row["bytes_sent"] for row in rows]
    assert sent["local"] == [0, 0, 0]
    assert all(
        0 < jpeg < raw for jpeg, raw in zip(sent["offload_jpeg"], sent["offload"], strict=True)
    )
    assert len(agent.devices) == 1  # one model, loaded once


def test_a_job_applies_its_timeout_to_the_edge(workspace, servicer, edge_addr, monkeypatch):
    release = _block_worker(servicer, monkeypatch)
    agent = Agent("device-3", edge_addr)
    job = {**workspace, "variant": "offload", "indices": [0], "warmup": 0, "timeout_s": TIMEOUT_S}
    (row,) = agent.run(job)["rows"]
    release.set()
    # Given up after the job's timeout, not after the client's default of 2 s, and answered.
    assert not row["edge_up"]
    assert TIMEOUT_S * MS_PER_S <= row["rpc_ms"] < 2 * TIMEOUT_S * MS_PER_S
    assert row["correct"]


def test_a_local_job_needs_no_edge(workspace):
    agent = Agent("device-3", edge_addr=None)
    rows = agent.run({**workspace, "variant": "local", "indices": [0, 2], "warmup": 0})["rows"]
    assert [row["correct"] for row in rows] == [True, True]
    assert [row["rpc_ms"] for row in rows] == [0.0, 0.0]


def test_a_job_that_needs_the_edge_fails_at_once_if_there_is_none(workspace, monkeypatch):
    server, port = start_server(port=0, host="localhost")
    server.stop(STOP_GRACE_S)
    monkeypatch.setattr(device_client, "EDGE_WAIT_S", TIMEOUT_S)
    agent = Agent("device-3", f"localhost:{port}")
    with pytest.raises(RuntimeError, match="edge"):
        agent.run({**workspace, "variant": "offload", "indices": [0], "warmup": 0})


def test_the_agent_answers_one_line_per_job(workspace, edge_addr, monkeypatch):
    jobs = [
        {**workspace, "variant": "local", "indices": [0, 2], "warmup": 0},
        {**workspace, "variant": "offload", "indices": [3], "warmup": 0},
    ]
    monkeypatch.setenv("DEVICE_ID", "device-9")
    monkeypatch.setenv("EDGE_ADDR", edge_addr)
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps(job) + "\n" for job in jobs)))
    output = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)

    threads = torch.get_num_threads()
    try:
        device_client.main()
        assert torch.get_num_threads() == 1  # a device has one core
    finally:
        torch.set_num_threads(threads)

    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [len(reply["rows"]) for reply in replies] == [2, 1]
    assert replies[1]["rows"][0]["device_id"] == "device-9"
    assert replies[1]["rows"][0]["variant"] == "offload"


@pytest.fixture
def exit_workspace(exit_model, workspace, tmp_path):
    """`workspace` with a checkpoint that has exit heads, as scripts/train_exits.py writes it."""
    checkpoint = tmp_path / "exits.pt"
    torch.save(
        {
            "model": exit_model.name,
            "state_dict": exit_model.state_dict(),
            "exits": {
                "blocks": EXITS[exit_model.name],
                "pool": HEAD_POOL,
                "hidden": HEAD_HIDDEN,
            },
        },
        checkpoint,
    )
    return {**workspace, "checkpoint": str(checkpoint)}


def test_a_pinned_job_serves_the_plan_it_names(exit_workspace, edge_addr, exit_model):
    agent = Agent("device-3", edge_addr)
    n = exit_model.num_blocks
    job = {**exit_workspace, "variant": "pinned", "indices": [0, 2, 3], "warmup": 0}
    rows = agent.run({**job, "k": n, "tau": TAU_ALWAYS})["rows"]
    assert {(row["variant"], row["k"], row["tau"]) for row in rows} == {("pinned", n, TAU_ALWAYS)}
    assert all(row["exited_on_device"] and row["bytes_sent"] == 0 for row in rows)

    k = exit_model.exit_points[0] + 1
    rows = agent.run({**job, "k": k, "tau": TAU_ALMOST_NEVER})["rows"]
    assert not any(row["exited_on_device"] for row in rows)
    assert all(row["bytes_sent"] > 0 and row["edge_up"] for row in rows)


@pytest.mark.parametrize(
    ("change", "message"),
    [({"k": None}, "k"), ({"k": 99}, "k"), ({"tau": 0.0}, "tau"), ({"tau": 1.5}, "tau")],
)
def test_a_pinned_job_needs_a_split_point_and_a_threshold(workspace, change, message):
    job = {**workspace, "variant": "pinned", "indices": [0], "warmup": 0, "k": 5, "tau": 0.9}
    with pytest.raises(ValueError, match=message):
        Agent("device-3", edge_addr=None).run({**job, **change})


def test_a_reply_says_what_the_device_believes_about_the_link(workspace, edge_addr):
    agent = Agent("device-3", edge_addr)
    job = {**workspace, "variant": "offload", "indices": [0, 2, 3, 4], "warmup": 0}
    reply = agent.run({**job, "rate_hz": 5.0})  # 200 ms between arrivals: room for probes
    assert set(reply["link"]) == {"bw_mbps", "rtt_ms", "queue_len"}
    assert reply["link"]["rtt_ms"] > 0 and reply["link"]["bw_mbps"] > 0
    assert reply["rows"][-1]["bw_est_mbps"] > 0
    json.dumps(reply)
