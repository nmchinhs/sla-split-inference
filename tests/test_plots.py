"""Summary and baseline figure (scripts/plots.py), on made-up runs with known statistics."""

import importlib.util
import json
from pathlib import Path

import pytest

from edgeinf.common.logging import RequestLog

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "plots.py"
spec = importlib.util.spec_from_file_location("plots", SCRIPT)
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)

N_BLOCKS = 20  # MobileNetV2: k = 20 is local
RATE_HZ = 5  # one request every 200 ms
RAW_IMAGE_BYTES = 150_550
PNG_MAGIC = b"\x89PNG"


def _row(variant: str, latency_ms: float, wait_ms: float = 0.0, **changes) -> dict:
    local = variant == "local"
    row = {
        "timestamp": 1_791_450_000.0,
        "device_id": "device-1",
        "variant": variant,
        "k": N_BLOCKS if local else 0,
        "tau": 1.0,
        "exited_on_device": False,
        "latency_ms": latency_ms,
        "bytes_sent": 0 if local else RAW_IMAGE_BYTES,
        "correct": True,
        "bw_est_mbps": None,
        "edge_queue_len": None if local else 0,
        "edge_up": True,
        "wait_ms": wait_ms,
        "device_ms": latency_ms - wait_ms if local else 1.0,
        "rpc_ms": 0.0 if local else latency_ms - wait_ms - 1.0,
        "edge_queue_ms": None if local else 0.1,
        "edge_compute_ms": None if local else 3.0,
    }
    return row | changes


def _write_run(folder: Path, trace: str, rate_mbit: float, repeat: int, rows: list[dict]) -> None:
    stem = folder / f"{trace}_r{repeat}"
    with RequestLog(stem.with_suffix(".csv")) as log:
        for row in rows:
            log.write(row)
    record = {
        "config": {"model": "mobilenetv2", "rate_hz": RATE_HZ},
        "trace": {"name": trace, "rate_mbit": rate_mbit, "delay_ms": 20.0, "loss_pct": 0.0},
        "repeat": repeat,
        "blocks": [{"variant": "local", "n_blocks": N_BLOCKS}],
    }
    stem.with_suffix(".json").write_text(json.dumps(record))


@pytest.fixture
def results(tmp_path) -> Path:
    """Two repeats at 5 Mbps and one at 50 Mbps, with statistics that can be checked by hand."""
    folder = tmp_path / "baselines_mobilenetv2"
    # 5 Mbps, repeat 0: local has median 100 and p95 130. offload takes 264 ms a request, more
    # than the 200 ms between arrivals, so each request waits 64 ms longer than the one before.
    _write_run(
        folder,
        "static_5mbps",
        5.0,
        0,
        [_row("local", v) for v in (90.0, 100.0, 130.0)]
        + [_row("offload", 264.0 + 64.0 * i, wait_ms=64.0 * i) for i in range(3)],
    )
    # 5 Mbps, repeat 1: local has median 110 and p95 111; one wrong answer.
    _write_run(
        folder,
        "static_5mbps",
        5.0,
        1,
        [_row("local", 105.0), _row("local", 110.0, correct=False), _row("local", 111.0)]
        + [_row("offload", 270.0 + 70.0 * i, wait_ms=70.0 * i) for i in range(3)],
    )
    # 50 Mbps: offload is fast; one of its requests fell back, and so did not reach the edge.
    _write_run(
        folder,
        "static_50mbps",
        50.0,
        0,
        [_row("local", 96.0, edge_up=False), _row("local", 98.0)]
        + [_row("offload", 47.0), _row("offload", 150.0, edge_up=False, edge_queue_len=None)],
    )
    (folder / "static_1mbps_r0.csv.part").write_text("an interrupted run\n")
    return folder


def _cell(summary, rate_mbit: float, variant: str) -> dict:
    rows = summary[(summary["rate_mbit"] == rate_mbit) & (summary["variant"] == variant)]
    assert len(rows) == 1
    return rows.iloc[0].to_dict()


def test_runs_are_loaded_with_what_their_records_say(results):
    requests = plots.load_runs(results)
    assert len(requests) == 6 + 6 + 4  # the interrupted run is not read
    assert set(requests["model"]) == {"mobilenetv2"}
    assert set(requests["trace"]) == {"static_5mbps", "static_50mbps"}
    assert sorted(set(requests["rate_mbit"])) == [5.0, 50.0]
    assert sorted(set(requests["repeat"])) == [0, 1]
    assert set(requests["rate_hz"]) == {RATE_HZ}
    assert set(requests["n_blocks"]) == {N_BLOCKS}


def test_a_folder_without_runs_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no runs"):
        plots.load_runs(tmp_path)


def test_latency_is_summarised_per_repeat_and_then_over_the_repeats(results):
    local = _cell(plots.summarise(plots.load_runs(results)), 5.0, "local")
    assert local["model"] == "mobilenetv2" and local["trace"] == "static_5mbps"
    assert local["repeats"] == 2 and local["requests"] == 6
    assert local["median_ms"] == 105.0  # mean of the medians 100 and 110
    assert (local["median_min_ms"], local["median_max_ms"]) == (100.0, 110.0)
    assert local["p95_ms"] == 120.5  # mean of 130 and 111
    assert (local["p95_min_ms"], local["p95_max_ms"]) == (111.0, 130.0)
    assert local["service_ms"] == 105.0  # it never waited
    assert not local["overloaded"]


def test_a_variant_that_cannot_keep_up_is_flagged_by_its_service_time(results):
    summary = plots.summarise(plots.load_runs(results))
    slow = _cell(summary, 5.0, "offload")
    # Latency grows with the queue, the service time does not: 264 and 270 ms, mean 267.
    assert slow["median_ms"] == (328.0 + 340.0) / 2
    assert slow["service_ms"] == 267.0
    assert slow["utilisation"] == pytest.approx(267.0 / 200.0)  # work per time between arrivals
    assert slow["overloaded"]
    fast = _cell(summary, 50.0, "offload")
    assert fast["utilisation"] == pytest.approx((47.0 + 150.0) / 2 / 200.0)
    assert not fast["overloaded"]


def test_overload_is_judged_by_the_mean_service_time_not_the_median(tmp_path):
    # Nine fast requests and one that took 5 s (a timeout): the median says 50 ms, but on
    # average a request takes 545 ms, far more than the 200 ms between arrivals.
    folder = tmp_path / "run"
    rows = [_row("offload", 50.0) for _ in range(9)] + [_row("offload", 5000.0)]
    _write_run(folder, "static_5mbps", 5.0, 0, rows)
    cell = _cell(plots.summarise(plots.load_runs(folder)), 5.0, "offload")
    assert cell["service_ms"] == 50.0
    assert cell["utilisation"] == pytest.approx(545.0 / 200.0)
    assert cell["overloaded"]


def test_back_to_back_runs_have_no_utilisation_and_are_never_overloaded(tmp_path):
    folder = tmp_path / "run"
    _write_run(folder, "static_50mbps", 50.0, 0, [_row("local", 5000.0)])
    record_path = folder / "static_50mbps_r0.json"
    record = json.loads(record_path.read_text())
    record["config"]["rate_hz"] = None
    record_path.write_text(json.dumps(record))
    cell = _cell(plots.summarise(plots.load_runs(folder)), 50.0, "local")
    assert cell["utilisation"] != cell["utilisation"]  # not a number
    assert not cell["overloaded"]


def test_the_summary_says_where_the_time_of_a_request_goes(results):
    summary = plots.summarise(plots.load_runs(results))
    slow = _cell(summary, 5.0, "offload")
    # Per repeat: 1 ms on the device, 3.1 ms at the edge (0.1 waiting, 3 computing), and the
    # rest of the 264 or 270 ms of service on the network.
    assert slow["device_ms"] == pytest.approx(1.0)
    assert slow["edge_ms"] == pytest.approx(3.1)
    assert slow["network_ms"] == pytest.approx((259.9 + 265.9) / 2)
    assert slow["device_ms"] + slow["network_ms"] + slow["edge_ms"] == pytest.approx(
        slow["service_ms"]
    )
    # A local request has no network and no edge part.
    local = _cell(summary, 5.0, "local")
    assert local["device_ms"] == pytest.approx(105.0)
    assert local["network_ms"] != local["network_ms"] and local["edge_ms"] != local["edge_ms"]


def test_accuracy_bytes_and_fallbacks_are_counted_over_all_requests(results):
    summary = plots.summarise(plots.load_runs(results))
    local = _cell(summary, 5.0, "local")
    assert local["accuracy"] == pytest.approx(100 * 5 / 6)
    assert local["kb_sent"] == 0.0
    assert _cell(summary, 5.0, "offload")["kb_sent"] == pytest.approx(RAW_IMAGE_BYTES / 1024)

    # A local request (k = n) never needs the edge, so it cannot fall back, whatever edge_up says.
    assert _cell(summary, 50.0, "local")["fallbacks"] == 0
    assert _cell(summary, 50.0, "offload")["fallbacks"] == 1
    assert _cell(summary, 5.0, "offload")["fallbacks"] == 0


def test_summary_has_one_row_per_model_bandwidth_and_variant_in_a_fixed_order(results):
    summary = plots.summarise(plots.load_runs(results))
    assert list(zip(summary["rate_mbit"], summary["variant"], strict=True)) == [
        (5.0, "local"),
        (5.0, "offload"),
        (50.0, "local"),
        (50.0, "offload"),
    ]


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 21)]
    assert plots.percentile(values, 0.95) == 19.0
    assert plots.percentile([3.0, 1.0, 2.0], 0.95) == 3.0
    assert plots.percentile([7.0], 0.95) == 7.0


def test_every_variant_of_the_project_has_its_own_colour():
    variants = ["local", "offload", "offload_jpeg", "split_only", "exit_only", "joint"]
    assert list(plots.VARIANT_COLOURS) == variants  # the order of CLAUDE.md
    assert len(set(plots.VARIANT_COLOURS.values())) == len(variants)


def test_the_baseline_figure_is_written(results, tmp_path):
    summary = plots.summarise(plots.load_runs(results))
    path = tmp_path / "figures" / "baselines_latency.png"
    plots.plot_baselines(summary, path)
    assert path.read_bytes()[:4] == PNG_MAGIC


def test_panels_show_every_bandwidth_when_the_models_have_different_ones(
    results, tmp_path, monkeypatch
):
    # The panels share the x axis, so the limits and ticks of one panel are those of all: a
    # model with fewer bandwidths (an unfinished run) must not cut off the other model's points.
    summary = plots.summarise(plots.load_runs(results))
    unfinished = summary[summary["rate_mbit"] == 5.0].assign(model="resnet18")
    figures = []
    monkeypatch.setattr(plots.plt, "close", figures.append)
    plots.plot_baselines(plots.pd.concat([summary, unfinished]), tmp_path / "figure.png")
    (figure,) = figures
    assert len(figure.axes) == 4
    for axes in figure.axes:
        low, high = axes.get_xlim()
        assert low < 5.0 and high > 50.0
        assert list(axes.get_xticks()) == [5.0, 50.0]
