"""Edge gRPC server: runs blocks [k, n) of a model on what a device sends.

Requests are served one at a time, in the order they arrive. The gRPC threads only receive a
request, put it in a queue and wait; one worker thread takes the requests from the queue and is
the only thread that touches the models and the GPU. That makes the three numbers of a reply
well defined:

    queue_len    Infer requests at the edge (waiting or in service) when this one arrived
    queue_ms     from its arrival until the worker started on it
    compute_ms   decoding the tensor and running blocks [k, n), up to label and confidence

Health is answered by the gRPC thread itself, so it never waits behind an inference; it is also
the device's link probe (it receives and discards padding).
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import threading
import time
from collections.abc import Iterable
from concurrent import futures
from dataclasses import dataclass, field

import grpc
import torch

from edgeinf.common.grpc_options import DEFAULT_EDGE_PORT, GRPC_OPTIONS
from edgeinf.common.serialize import deserialize
from edgeinf.data.imagenette import normalise
from edgeinf.models.multi_exit import INPUT_SIZE, MultiExitModel, load_checkpoint
from edgeinf.models.split import run_tail, top1
from edgeinf.rpc import inference_pb2 as pb
from edgeinf.rpc import inference_pb2_grpc as pb_grpc

# gRPC threads. More than ten devices with an Infer and a Health call each, so that no request
# waits where queue_len and queue_ms would not see it.
MAX_WORKERS = 32
SHUTDOWN_GRACE_S = 1.0
MS_PER_S = 1000.0
CHECKPOINTS_ENV = "EDGE_CHECKPOINTS"  # comma-separated checkpoint files to serve

log = logging.getLogger(__name__)

# What a block takes on the wire: dtype and full shape, batch of one.
BlockInput = tuple[torch.dtype, tuple[int, ...]]


@dataclass
class _Job:
    """One Infer request on its way through the queue."""

    model: MultiExitModel
    data: bytes
    k: int
    done: threading.Event = field(default_factory=threading.Event)
    started: float = 0.0  # time.perf_counter() when the worker took it
    finished: float = 0.0
    answer: tuple[int, float] | None = None  # label, confidence
    error: tuple[grpc.StatusCode, str] | None = None


def _block_inputs(model: MultiExitModel, device: str) -> list[BlockInput]:
    """For k = 0..n-1, the dtype and shape of what a device sends for split point k.

    Found by running one frame through the model, which also makes the first real request find
    every layer already initialised on the GPU.
    """
    frame = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.uint8, device=device)
    inputs: list[BlockInput] = [(frame.dtype, tuple(frame.shape))]
    x = normalise(frame)
    for block in model.blocks[:-1]:
        x = block(x)
        inputs.append((x.dtype, tuple(x.shape)))
    model.blocks[-1](x)
    return inputs


class InferenceServicer(pb_grpc.InferenceServicer):
    """Serves Infer and Health for `models`, which it moves to `device` ("cuda" or "cpu")."""

    def __init__(self, models: Iterable[MultiExitModel] = (), device: str = "cpu") -> None:
        # TF32 would make a GPU edge compute something else than the CPU devices do.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self._device = device
        self._models = {model.name: model.to(device).eval() for model in models}
        with torch.inference_mode():
            self._inputs = {
                name: _block_inputs(model, device) for name, model in self._models.items()
            }
        self._lock = threading.Lock()
        self._in_flight = 0
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._work, name="inference", daemon=True)
        self._worker.start()

    def close(self) -> None:
        """Stop the worker thread after the requests already queued."""
        self._jobs.put(None)
        self._worker.join()

    def queue_len(self) -> int:
        """Number of Infer requests currently at the edge (waiting or in service)."""
        with self._lock:
            return self._in_flight

    def block_inputs(self, model: str) -> list[BlockInput]:
        """For k = 0..n-1, the dtype and shape of the tensor that `model` takes at split point k."""
        return self._inputs[model]

    def _answer(self, model: MultiExitModel, data: bytes, k: int) -> tuple[int, float]:
        """Label and confidence for the bytes of split point k. ValueError if they do not fit."""
        x = deserialize(data)
        dtype, shape = self._inputs[model.name][k]
        if x.dtype != dtype or tuple(x.shape) != shape:
            raise ValueError(
                f"{model.name} takes {dtype} {shape} at k = {k}, got {x.dtype} {tuple(x.shape)}"
            )
        return top1(run_tail(model, x.to(self._device), k))

    def _work(self) -> None:
        """The worker thread: serve queued jobs one at a time, in order."""
        with torch.inference_mode():
            while (job := self._jobs.get()) is not None:
                job.started = time.perf_counter()
                try:
                    job.answer = self._answer(job.model, job.data, job.k)
                except ValueError as error:  # the request's fault: bad bytes or a wrong tensor
                    job.error = (grpc.StatusCode.INVALID_ARGUMENT, str(error))
                except Exception as error:  # the edge's fault; keep serving the next request
                    log.exception("inference failed")
                    job.error = (grpc.StatusCode.INTERNAL, f"{type(error).__name__}: {error}")
                job.finished = time.perf_counter()
                job.done.set()

    def Infer(self, request: pb.InferRequest, context: grpc.ServicerContext) -> pb.InferReply:
        arrived = time.perf_counter()
        with self._lock:
            ahead = self._in_flight
            self._in_flight += 1
        try:
            model = self._models.get(request.meta.model)
            if model is None:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unknown model {request.meta.model!r}; serving {sorted(self._models)}",
                )
            if request.k >= model.num_blocks:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"k = {request.k} is outside [0, {model.num_blocks}) for {model.name}",
                )
            job = _Job(model, request.tensor, request.k)
            self._jobs.put(job)
            job.done.wait()
            if job.error is not None:
                context.abort(*job.error)
            label, confidence = job.answer
            return pb.InferReply(
                label=label,
                confidence=confidence,
                queue_len=ahead,
                queue_ms=(job.started - arrived) * MS_PER_S,
                compute_ms=(job.finished - job.started) * MS_PER_S,
            )
        finally:
            with self._lock:
                self._in_flight -= 1

    def Health(self, request: pb.HealthRequest, context: grpc.ServicerContext) -> pb.HealthReply:
        return pb.HealthReply(
            ok=True, queue_len=self.queue_len(), padding_bytes=len(request.padding)
        )


def start_server(
    port: int = DEFAULT_EDGE_PORT,
    host: str = "[::]",
    servicer: InferenceServicer | None = None,
) -> tuple[grpc.Server, int]:
    """Start the server and return it with the bound port (pass port=0 to pick a free one).

    Without a `servicer` the server has no models: it answers Health and refuses every Infer.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS), options=GRPC_OPTIONS)
    pb_grpc.add_InferenceServicer_to_server(servicer or InferenceServicer(), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server, bound_port


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [path for path in os.environ.get(CHECKPOINTS_ENV, "").split(",") if path]
    # The port opens only when the models are loaded, so a device that reaches the edge can use it.
    servicer = InferenceServicer([load_checkpoint(path) for path in paths], device)
    server, port = start_server(
        int(os.environ.get("EDGE_PORT", DEFAULT_EDGE_PORT)), servicer=servicer
    )
    served = ", ".join(sorted(servicer._models)) or "no models"
    print(f"edge server listening on port {port}, serving {served} on {device}", flush=True)
    # As PID 1 in the container, SIGTERM is ignored unless handled, so `docker stop` would
    # hang until its kill timeout.
    signal.signal(signal.SIGTERM, lambda *_: server.stop(SHUTDOWN_GRACE_S))
    server.wait_for_termination()


if __name__ == "__main__":
    main()
