"""Per-request CSV log (common/logging.py): the column list is a contract with the report."""

import csv

import pytest

from edgeinf.common.logging import FIELDS, SCHEMA, TIMING, RequestLog

# Written out here on purpose: CLAUDE.md fixes these names and their order, and the analysis
# scripts and the report depend on them. Changing them must fail a test.
SCHEMA_IN_CLAUDE_MD = [
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
]
TIMING_COLUMNS = ["wait_ms", "device_ms", "rpc_ms", "edge_queue_ms", "edge_compute_ms"]

ROW = {
    "timestamp": 1791450000.1234567,
    "device_id": "device-1",
    "variant": "offload",
    "k": 0,
    "tau": 1.0,
    "exited_on_device": False,
    "latency_ms": 47.123456,
    "bytes_sent": 150550,
    "correct": True,
    "bw_est_mbps": None,
    "edge_queue_len": 2,
    "edge_up": True,
    "wait_ms": 0.0,
    "device_ms": 0.25,
    "rpc_ms": 46.873456,
    "edge_queue_ms": 0.0312,
    "edge_compute_ms": 3.1,
}


def _read(path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_columns_are_the_schema_followed_by_the_timing_columns():
    assert list(SCHEMA) == SCHEMA_IN_CLAUDE_MD
    assert list(TIMING) == TIMING_COLUMNS
    assert list(FIELDS) == SCHEMA_IN_CLAUDE_MD + TIMING_COLUMNS


def test_header_is_written_even_without_rows(tmp_path):
    path = tmp_path / "run.csv"
    with RequestLog(path):
        pass
    assert path.read_text().splitlines() == [",".join(SCHEMA_IN_CLAUDE_MD + TIMING_COLUMNS)]


def test_one_line_per_request_in_the_order_written(tmp_path):
    path = tmp_path / "run.csv"
    with RequestLog(path) as log:
        for latency_ms in (10.0, 30.0, 20.0):
            log.write({**ROW, "latency_ms": latency_ms})
    rows = _read(path)
    assert [float(row["latency_ms"]) for row in rows] == [10.0, 30.0, 20.0]
    assert list(rows[0]) == SCHEMA_IN_CLAUDE_MD + TIMING_COLUMNS


def test_values_are_written_as_numbers_with_flags_as_0_or_1_and_unknowns_empty(tmp_path):
    path = tmp_path / "run.csv"
    with RequestLog(path) as log:
        log.write(ROW)
        log.write({**ROW, "exited_on_device": True, "correct": False, "edge_up": False})
    first, second = _read(path)

    assert (first["exited_on_device"], first["correct"], first["edge_up"]) == ("0", "1", "1")
    assert (second["exited_on_device"], second["correct"], second["edge_up"]) == ("1", "0", "0")
    assert first["bw_est_mbps"] == ""  # no estimate: empty, not 0 and not "None"
    assert first["device_id"] == "device-1" and first["variant"] == "offload"
    assert (first["k"], first["bytes_sent"], first["edge_queue_len"]) == ("0", "150550", "2")
    # Times keep microsecond resolution: three decimals of a millisecond, six of a second.
    assert first["latency_ms"] == "47.123"
    assert first["edge_queue_ms"] == "0.031"
    assert first["timestamp"] == "1791450000.123457"
    assert float(first["tau"]) == 1.0


@pytest.mark.parametrize("missing", ["latency_ms", "edge_up", "wait_ms"])
def test_a_row_with_a_missing_column_is_refused(tmp_path, missing):
    row = {name: value for name, value in ROW.items() if name != missing}
    with RequestLog(tmp_path / "run.csv") as log, pytest.raises(ValueError, match=missing):
        log.write(row)


def test_a_row_with_an_unknown_column_is_refused(tmp_path):
    with RequestLog(tmp_path / "run.csv") as log, pytest.raises(ValueError, match="edge_ms"):
        log.write({**ROW, "edge_ms": 3.0})


def test_rows_are_on_disk_as_soon_as_they_are_written(tmp_path):
    path = tmp_path / "run.csv"
    with RequestLog(path) as log:
        log.write(ROW)
        assert len(_read(path)) == 1  # a run that dies later has lost nothing


def test_missing_folders_are_created(tmp_path):
    path = tmp_path / "results" / "name" / "run.csv"
    with RequestLog(path) as log:
        log.write(ROW)
    assert len(_read(path)) == 1
