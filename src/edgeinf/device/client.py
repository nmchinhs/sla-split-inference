"""Device runtime: serves requests on the device, with the edge's help where the plan says so.

    EdgeClient   the two RPCs to the edge, each with a timeout
    Plan         how one request is served: variant, split point k, threshold tau
    Device       serves one request: blocks [0, k), send, wait for the answer, or finish alone
    serve        the arrival loop: requests at a fixed rate, one row per request
    Agent, main  the process that runs in a device container and takes jobs from the host

A device serves one request at a time, and waiting for the edge counts as busy. A request that
arrives meanwhile waits and is never dropped; its latency runs from its scheduled arrival.

Fallback: if the edge does not answer an Infer in time, answers with an error, or is believed to
be down and a Health probe does not say otherwise, the device runs the remaining blocks itself.
Every fallback is logged.

Must stay free of CUDA calls: it runs in the CPU-only image. Exit checks come with Phase 3.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import grpc
import numpy as np
import torch

from edgeinf.common.grpc_options import GRPC_OPTIONS
from edgeinf.data.imagenette import load_frames
from edgeinf.models.multi_exit import MultiExitModel, load_checkpoint
from edgeinf.models.split import pack, run_head, run_tail
from edgeinf.rpc import inference_pb2 as pb
from edgeinf.rpc import inference_pb2_grpc as pb_grpc

DEFAULT_TIMEOUT_S = 2.0
HEALTH_TIMEOUT_S = 1.0  # for the probe before an edge believed to be down is used again
EDGE_WAIT_S = 60.0  # how long the agent waits for the edge before its first job that needs it
EDGE_POLL_S = 0.5
DEVICE_THREADS = 1  # a device has one core
MS_PER_S = 1e3

# The variants whose plan is fixed. The others (split_only, exit_only, joint) need the scheduler.
VARIANTS = ("local", "offload", "offload_jpeg")
NO_EXITS = 1.0  # tau at which no exit is ever taken

log = logging.getLogger(__name__)


class EdgeClient:
    """Talks to the edge at `addr` ("host:port"). Every call has a timeout in seconds."""

    def __init__(self, addr: str, device_id: str = "", timeout_s: float = DEFAULT_TIMEOUT_S):
        self.addr = addr
        self.device_id = device_id
        self.timeout_s = timeout_s
        self._channel = grpc.insecure_channel(addr, options=GRPC_OPTIONS)
        self._stub = pb_grpc.InferenceStub(self._channel)

    def health(self, padding: bytes = b"", timeout_s: float | None = None) -> pb.HealthReply | None:
        """Return the edge's health reply, or None if it is unreachable or timed out.

        `padding` is discarded by the edge. Timing the call with it empty measures RTT; with a
        known number of bytes it measures uplink bandwidth without running inference.
        """
        request = pb.HealthRequest(padding=padding)
        try:
            return self._stub.Health(request, timeout=timeout_s or self.timeout_s)
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


@dataclass(frozen=True)
class Plan:
    """How a request is served: the device runs blocks [0, k) and sends the rest to the edge."""

    variant: str
    k: int
    tau: float = NO_EXITS
    jpeg_quality: int | None = None  # send the frame as a JPEG of this quality (k = 0 only)


def fixed_plan(variant: str, n_blocks: int, jpeg_quality: int | None = None) -> Plan:
    """The plan of a variant that never changes it: local, offload or offload_jpeg.

    `jpeg_quality` (1 to 100) is used by offload_jpeg only.
    """
    if variant == "local":
        return Plan(variant, n_blocks)
    if variant == "offload":
        return Plan(variant, 0)
    if variant == "offload_jpeg":
        if jpeg_quality is None:
            raise ValueError("offload_jpeg needs a jpeg_quality")
        return Plan(variant, 0, jpeg_quality=jpeg_quality)
    raise ValueError(f"variant {variant!r} has no fixed plan; expected one of {VARIANTS}")


@dataclass
class Answer:
    """The outcome of one request, as the device saw it. Times in ms."""

    label: int
    exited_on_device: bool  # an exit head answered
    bytes_sent: int  # size of the tensor handed to gRPC, 0 if none was
    edge_up: bool  # False: the edge was unavailable and the device finished alone
    rpc_ms: float  # time spent waiting for the edge: Infer, and a Health probe if one was made
    edge_queue_len: int | None = None  # from the edge's answer; None without one
    edge_queue_ms: float | None = None
    edge_compute_ms: float | None = None


class Device:
    """Serves requests with `model`, offloading through `client` (None: there is no edge).

    `edge_up` is what the device believes about the edge. It starts as true, turns false when an
    Infer fails, and true again when a Health probe is answered with ok.
    """

    def __init__(self, model: MultiExitModel, client: EdgeClient | None) -> None:
        self.model = model
        self.client = client
        self.edge_up = client is not None
        self._requests = 0

    def infer(self, frame: torch.Tensor, plan: Plan) -> Answer:
        """Serve one 8-bit frame, shape (1, 3, H, W), as `plan` says. Never raises for the edge.

        The caller disables gradients.
        """
        request_id, self._requests = self._requests, self._requests + 1
        model = self.model
        x = run_head(model, frame, plan.k)
        if plan.k == model.num_blocks:
            return Answer(int(x.argmax(1)), False, bytes_sent=0, edge_up=self.edge_up, rpc_ms=0.0)

        bytes_sent, waited_s, reason = 0, 0.0, "no edge is configured"
        if self.client is not None and not self.edge_up:
            start = time.perf_counter()
            health = self.client.health(timeout_s=HEALTH_TIMEOUT_S)
            waited_s += time.perf_counter() - start
            self.edge_up = health is not None and health.ok
            reason = "the edge is still unavailable (Health probe)"
        if self.client is not None and self.edge_up:
            payload = pack(x, plan.k, plan.jpeg_quality)
            bytes_sent = len(payload)
            start = time.perf_counter()
            try:
                reply = self.client.infer(payload, plan.k, request_id, model.name)
            except grpc.RpcError as error:
                waited_s += time.perf_counter() - start
                self.edge_up = False
                reason = f"Infer failed with {error.code().name} ({error.details()})"
            else:
                waited_s += time.perf_counter() - start
                return Answer(
                    reply.label,
                    False,
                    bytes_sent,
                    edge_up=True,
                    rpc_ms=waited_s * MS_PER_S,
                    edge_queue_len=reply.queue_len,
                    edge_queue_ms=reply.queue_ms,
                    edge_compute_ms=reply.compute_ms,
                )

        log.warning(
            "fallback: request %d (%s, k = %d) is finished on the device: %s",
            request_id,
            plan.variant,
            plan.k,
            reason,
        )
        label = int(run_tail(model, x, plan.k).argmax(1))
        return Answer(label, False, bytes_sent, edge_up=False, rpc_ms=waited_s * MS_PER_S)


def serve(
    device: Device,
    frames: torch.Tensor,
    labels: Sequence[int],
    plan: Plan,
    *,
    rate_hz: float | None,
    warmup: int,
    device_id: str,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    wall: Callable[[], float] = time.time,
) -> list[dict]:
    """Serve one request per frame, arriving `rate_hz` times per second, and return their rows.

    Request i is due at start + i / rate_hz. The device serves one request at a time, so a
    request starts when it is due or when the previous one has finished, whichever is later, and
    none is dropped. Its latency runs from when it was due. With `rate_hz` None the requests are
    back to back: each is due when the previous one finishes.

    `frames` is uint8, shape (N, 3, H, W), with one true label each. The first `warmup` requests
    are served but get no row. A row has the columns of common/logging.py. `clock` and `sleep`
    count seconds, `wall` is Unix time.
    """
    if rate_hz is not None and rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, or None for back to back; got {rate_hz}")
    rows = []
    start, wall_start = clock(), wall()
    for i, label in enumerate(labels):
        due = clock() if rate_hz is None else start + i / rate_hz
        if (idle_s := due - clock()) > 0:
            sleep(idle_s)
        started = clock()
        answer = device.infer(frames[i : i + 1], plan)
        finished = clock()
        if i < warmup:
            continue
        row = {
            "timestamp": wall_start + (due - start),
            "device_id": device_id,
            "variant": plan.variant,
            "k": plan.k,
            "tau": plan.tau,
            "exited_on_device": answer.exited_on_device,
            "latency_ms": (finished - due) * MS_PER_S,
            "bytes_sent": answer.bytes_sent,
            "correct": answer.label == int(label),
            "bw_est_mbps": None,  # the estimator comes with the scheduler
            "edge_queue_len": answer.edge_queue_len,
            "edge_up": answer.edge_up,
            "wait_ms": (started - due) * MS_PER_S,
            "device_ms": (finished - started) * MS_PER_S - answer.rpc_ms,
            "rpc_ms": answer.rpc_ms,
            "edge_queue_ms": answer.edge_queue_ms,
            "edge_compute_ms": answer.edge_compute_ms,
        }
        rows.append(row)
    return rows


class Agent:
    """The process in a device container: keeps models and frames loaded and runs jobs.

    A job is a dict:
        checkpoint    file written by scripts/train_backbone.py
        frames_dir    folder with the exported frames (scripts/export_frames.py)
        split         "val" or "test"
        indices       which frames to serve, in this order
        variant       local, offload or offload_jpeg
        jpeg_quality  for offload_jpeg
        rate_hz       requests per second, or null for back to back
        warmup        how many of the first requests get no row
        timeout_s     timeout of an Infer call
    The reply is {"rows": one dict per logged request, "minor_faults": page faults while
    serving, "n_blocks": n of the model, the k that means all local}.
    """

    def __init__(self, device_id: str, edge_addr: str | None) -> None:
        self.device_id = device_id
        self.client = EdgeClient(edge_addr, device_id) if edge_addr else None
        self.devices: dict[str, Device] = {}  # by checkpoint
        self._frames: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._edge_seen = False

    def _await_edge(self) -> None:
        """Wait until the edge says it is healthy; only before the first job that needs it."""
        if self._edge_seen:
            return
        if self.client is None:
            raise RuntimeError("the job needs the edge, but EDGE_ADDR is not set")
        deadline = time.monotonic() + EDGE_WAIT_S
        while True:
            health = self.client.health(timeout_s=HEALTH_TIMEOUT_S)
            if health is not None and health.ok:
                self._edge_seen = True
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"no healthy edge at {self.client.addr} after {EDGE_WAIT_S} s")
            time.sleep(EDGE_POLL_S)

    def run(self, job: dict) -> dict:
        """Serve the requests of one job and return its rows."""
        checkpoint = job["checkpoint"]
        if checkpoint not in self.devices:
            self.devices[checkpoint] = Device(load_checkpoint(checkpoint), self.client)
        device = self.devices[checkpoint]
        source = (job["frames_dir"], job["split"])
        if source not in self._frames:
            self._frames[source] = load_frames(*source)
        all_frames, all_labels = self._frames[source]
        indices = job["indices"]
        # Copies just these frames out of the memory-mapped file, before the clock starts.
        frames = torch.from_numpy(np.ascontiguousarray(all_frames[indices]))
        labels = [int(all_labels[index]) for index in indices]

        plan = fixed_plan(job["variant"], device.model.num_blocks, job.get("jpeg_quality"))
        if plan.k < device.model.num_blocks:
            self._await_edge()
        if self.client is not None:
            self.client.timeout_s = job["timeout_s"]
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        with torch.inference_mode():
            rows = serve(
                device,
                frames,
                labels,
                plan,
                rate_hz=job["rate_hz"],
                warmup=job["warmup"],
                device_id=self.device_id,
            )
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults
        return {"rows": rows, "minor_faults": faults, "n_blocks": device.model.num_blocks}


def main() -> None:
    """Run as the agent: one JSON job per line on stdin, one JSON reply per line on stdout."""
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s"
    )
    torch.set_num_threads(DEVICE_THREADS)
    agent = Agent(os.environ.get("DEVICE_ID", ""), os.environ.get("EDGE_ADDR"))
    for line in sys.stdin:
        print(json.dumps(agent.run(json.loads(line))), flush=True)


if __name__ == "__main__":
    main()
