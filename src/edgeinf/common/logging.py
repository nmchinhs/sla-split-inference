"""Per-request CSV log: one row per request.

The first twelve columns are the schema fixed in CLAUDE.md; the timing columns follow them and say
where the request's time went. All times are in ms.

    timestamp         Unix time in s at which the request was due (its scheduled arrival)
    device_id         name of the device that served it
    variant           local, offload, offload_jpeg, split_only, exit_only or joint
    k                 split point planned for the request (n: all local, 0: all offloaded)
    tau               exit threshold in force (1.0: exits are never taken)
    exited_on_device  1 if an exit head on the device answered
    latency_ms        from the scheduled arrival to the answer, so waiting is included
    bytes_sent        size in bytes of the tensor handed to gRPC; 0 if nothing was sent
    correct           1 if the predicted label is the true one
    bw_est_mbps       the device's bandwidth estimate in Mbps; empty while there is none
    edge_queue_len    queue length the edge reported with its answer; empty without an answer
    edge_up           0 if the device held the edge to be unavailable when the request finished,
                      which for k < n means the request was finished locally (fallback)

    wait_ms           from the scheduled arrival until the device started on the request
    device_ms         the device's own work on it: blocks, encoding, and the rest in a fallback
    rpc_ms            time the device waited for the edge (Infer, and a Health probe if one was
                      made); 0 if it did not contact the edge
    edge_queue_ms     waiting at the edge, from its answer; empty without an answer
    edge_compute_ms   decoding and blocks [k, n) at the edge, from its answer

latency_ms = wait_ms + device_ms + rpc_ms, and rpc_ms - edge_queue_ms - edge_compute_ms is the
time spent on the network.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType

SCHEMA = (
    "timestamp",
    "device_id",
    "variant",
    "k",
    "tau",
    "exited_on_device",
    "latency_ms",
    "bytes_sent",
    "correct",
    "bw_est_mbps",
    "edge_queue_len",
    "edge_up",
)
TIMING = ("wait_ms", "device_ms", "rpc_ms", "edge_queue_ms", "edge_compute_ms")
FIELDS = SCHEMA + TIMING

DECIMALS = 3  # of a float column: microseconds for the ms columns
TIMESTAMP_DECIMALS = 6  # microseconds of a Unix time in s


def _text(name: str, value: object) -> object:
    """A value as it is written: flags as 0 or 1, floats with fixed decimals.

    None (unknown) is passed on as it is: the csv module writes it as an empty field.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.{TIMESTAMP_DECIMALS if name == 'timestamp' else DECIMALS}f}"
    return value


class RequestLog:
    """Writes request rows to the CSV file at `path`, creating its folder and the header."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FIELDS)
        self._file.flush()

    def write(self, row: Mapping[str, object]) -> None:
        """Append one request. `row` must have exactly the columns in FIELDS; None means unknown."""
        if missing := [name for name in FIELDS if name not in row]:
            raise ValueError(f"request row lacks the columns {missing}")
        if unknown := sorted(set(row) - set(FIELDS)):
            raise ValueError(f"request row has unknown columns {unknown}")
        self._writer.writerow([_text(name, row[name]) for name in FIELDS])
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RequestLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
