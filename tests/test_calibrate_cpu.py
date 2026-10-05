"""The pure parts of scripts/calibrate_cpu.py: setting checks and the per-setting summary."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_cpu.py"
spec = importlib.util.spec_from_file_location("calibrate_cpu", SCRIPT)
calibrate_cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate_cpu)

QUOTA_US = 2_000
PERIOD_US = 10_000


def _row(repeat: int, latency_ms: float, pattern: str = "burst") -> dict:
    return {
        "quota_us": QUOTA_US,
        "period_us": PERIOD_US,
        "repeat": repeat,
        "model": "mobilenetv2",
        "k": 20,
        "pattern": pattern,
        "latency_ms": latency_ms,
        "cpu_ms": latency_ms / 5,
        "page_faults": 0,
    }


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 21)]  # 1..20
    assert calibrate_cpu.percentile(values, 0.95) == 19.0
    assert calibrate_cpu.percentile(values, 0.5) == 10.0
    assert calibrate_cpu.percentile(values, 1.0) == 20.0
    assert calibrate_cpu.percentile([7.0], 0.95) == 7.0


def test_percentile_does_not_depend_on_order():
    assert calibrate_cpu.percentile([3.0, 1.0, 2.0], 0.95) == 3.0


@pytest.mark.parametrize(
    ("quota_us", "period_us"),
    [(1_000, 4_000), (2_500, 10_000), (100_000, 100_000), (1_000, 1_000_000)],
)
def test_check_setting_accepts_what_docker_accepts(quota_us, period_us):
    calibrate_cpu.check_setting(quota_us, period_us)


@pytest.mark.parametrize(
    ("quota_us", "period_us"),
    [
        (999, 10_000),  # quota below Docker's minimum
        (1_000, 999),  # period below Docker's minimum
        (1_000, 1_000_001),  # period above Docker's maximum
        (20_000, 10_000),  # more than one CPU, but a device has one
    ],
)
def test_check_setting_rejects_what_docker_rejects(quota_us, period_us):
    with pytest.raises(ValueError):
        calibrate_cpu.check_setting(quota_us, period_us)


def test_summary_averages_per_repeat_medians_and_reports_their_spread():
    # Repeat 0 has median 100 ms and repeat 1 has median 110 ms.
    rows = [_row(0, v) for v in (90.0, 100.0, 130.0)] + [_row(1, v) for v in (105.0, 110.0, 111.0)]
    key = (QUOTA_US, PERIOD_US, "mobilenetv2", 20, "burst")

    (summary,) = calibrate_cpu.summarise(rows, {key: [0.19, 0.21]})

    assert summary["share"] == 0.2
    assert summary["repeats"] == 2
    assert summary["median_ms"] == 105.0
    assert (summary["median_min_ms"], summary["median_max_ms"]) == (100.0, 110.0)
    assert summary["p95_ms"] == (130.0 + 111.0) / 2
    assert summary["achieved_share"] == 0.2
    assert summary["page_faults"] == 0


def test_summary_keeps_patterns_apart_and_tolerates_a_missing_share():
    rows = [_row(0, 100.0, "burst"), _row(0, 80.0, "spaced")]

    summary = calibrate_cpu.summarise(rows, {})

    assert {s["pattern"]: s["median_ms"] for s in summary} == {"burst": 100.0, "spaced": 80.0}
    assert all(s["achieved_share"] == "" for s in summary)
