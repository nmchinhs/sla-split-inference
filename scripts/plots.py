"""Summaries and figures from the request CSVs that scripts/run_experiment.py writes to results/.

    python scripts/plots.py summary results/<name> [results/<name> ...]
    python scripts/plots.py baselines results/baselines_mobilenetv2 results/baselines_resnet18

`summary` prints one line per model, bandwidth and variant. `baselines` also writes that table to
experiments/baselines/summary.csv and the figure to figures/baselines_latency.png and .pdf.

How a cell is summarised. The median and the 95th percentile of the latency are taken per repeat
and then averaged over the repeats; the lowest and highest repeat give the spread. A request's
service time is its latency without the time it waited for the device to become free; it is
split into the device's own work, the time at the edge (waiting and computing there) and the
rest of the round trip, which is the network. These are medians over the requests that the edge
answered, per repeat and then averaged. If the
mean service time is longer than the time between two arrivals (utilisation above 1), the
variant cannot keep up at that rate: its queue grows for as long as requests keep arriving, so
its latency says how long the run was, not how fast the variant is. Such cells are flagged as
overloaded and drawn with hollow markers.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = REPO_ROOT / "experiments" / "baselines" / "summary.csv"
FIGURE_PATH = REPO_ROOT / "figures" / "baselines_latency"

P95 = 0.95
MS_PER_S = 1000.0
BYTES_PER_KB = 1024
PERCENT = 100.0
SUMMARY_DECIMALS = 2

# One colour per variant, the same in every figure. The order is the project's order of
# variants; the first three were checked for colour-blind separation as a set (all pairs).
VARIANT_COLOURS = {
    "local": "#2a78d6",  # blue
    "offload": "#eb6834",  # orange
    "offload_jpeg": "#1baf7a",  # aqua
    "split_only": "#eda100",  # yellow
    "exit_only": "#e87ba4",  # magenta
    "joint": "#008300",  # green
}
MODEL_TITLES = {"mobilenetv2": "MobileNetV2", "resnet18": "ResNet-18"}
STATISTICS = (("median", "Median latency (ms)"), ("p95", "95th-percentile latency (ms)"))

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LINE_WIDTH = 2.0
MARKER_SIZE = 7.5  # points
RING_WIDTH = 1.5  # points, in the surface colour around a marker
WHISKER_WIDTH = 1.0
FADED = 0.35  # opacity of a line segment that leads to an overloaded cell
PANEL_SIZE = (4.6, 3.1)  # inches, width and height of one panel
FIGURE_DPI = 200
LABEL_SIZE = 8
LABEL_OFFSET = 7  # points from the last marker to its label
LABEL_ROOM = 2.6  # the x axis extends to this multiple of the highest bandwidth, for the labels
LABEL_MIN_RATIO = 1.3  # two line ends closer than this factor get no labels: they would collide
OVERLOADED_NOTE = "hollow: cannot keep up with the arrivals; latency grows with the run's length"


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value with at least `fraction` of the data <= it."""
    ordered = sorted(values)
    return ordered[max(math.ceil(fraction * len(ordered)), 1) - 1]


def load_runs(results_dir: Path) -> pd.DataFrame:
    """All requests of the finished runs in `results_dir`, one row each.

    To the columns of the CSVs it adds, from each run's JSON record: model, trace, rate_mbit,
    repeat, rate_hz (requests per second; missing for back to back) and n_blocks.
    """
    runs = []
    for csv_path in sorted(results_dir.glob("*_r*.csv")):
        record = json.loads(csv_path.with_suffix(".json").read_text())
        requests = pd.read_csv(csv_path)
        requests["model"] = record["config"]["model"]
        requests["trace"] = record["trace"]["name"]
        requests["rate_mbit"] = float(record["trace"]["rate_mbit"])
        requests["repeat"] = record["repeat"]
        requests["rate_hz"] = record["config"]["rate_hz"]
        requests["n_blocks"] = record["blocks"][0]["n_blocks"]
        runs.append(requests)
    if not runs:
        raise FileNotFoundError(f"no runs in {results_dir}")
    return pd.concat(runs, ignore_index=True)


def summarise(requests: pd.DataFrame) -> pd.DataFrame:
    """One row per model, bandwidth and variant; see the module docstring for the statistics.

    Times are in ms, accuracy in percent, kb_sent in KB per request (1 KB = 1024 bytes).
    `fallbacks` counts requests that needed the edge and were finished on the device.
    """
    rows = []
    keys = ["model", "trace", "rate_mbit", "variant"]
    for (model, trace, rate_mbit, variant), cell in requests.groupby(keys):
        by_repeat = cell.groupby("repeat")["latency_ms"]
        medians = by_repeat.median()
        p95s = by_repeat.apply(lambda latencies: percentile(latencies.tolist(), P95))
        service = (cell["latency_ms"] - cell["wait_ms"]).groupby(cell["repeat"])
        at_edge = cell["edge_queue_ms"] + cell["edge_compute_ms"]  # missing without an answer
        on_network = cell["rpc_ms"] - at_edge
        rate_hz = cell["rate_hz"].iloc[0]
        utilisation = service.mean().mean() * rate_hz / MS_PER_S if pd.notna(rate_hz) else math.nan
        rows.append(
            {
                "model": model,
                "trace": trace,
                "rate_mbit": rate_mbit,
                "rate_hz": rate_hz,
                "variant": variant,
                "repeats": len(medians),
                "requests": len(cell),
                "median_ms": medians.mean(),
                "median_min_ms": medians.min(),
                "median_max_ms": medians.max(),
                "p95_ms": p95s.mean(),
                "p95_min_ms": p95s.min(),
                "p95_max_ms": p95s.max(),
                "service_ms": service.median().mean(),
                "device_ms": cell.groupby("repeat")["device_ms"].median().mean(),
                "network_ms": on_network.groupby(cell["repeat"]).median().mean(),
                "edge_ms": at_edge.groupby(cell["repeat"]).median().mean(),
                "utilisation": utilisation,
                "overloaded": bool(utilisation > 1),
                "accuracy": PERCENT * cell["correct"].mean(),
                "kb_sent": cell["bytes_sent"].mean() / BYTES_PER_KB,
                "fallbacks": int(((cell["edge_up"] == 0) & (cell["k"] < cell["n_blocks"])).sum()),
            }
        )
    variants = list(VARIANT_COLOURS)
    models = list(MODEL_TITLES)
    rows.sort(
        key=lambda row: (
            models.index(row["model"]),
            row["rate_mbit"],
            variants.index(row["variant"]),
        )
    )
    return pd.DataFrame(rows)


def _plain(value: float, _position: int) -> str:
    """A tick label without exponent: 100, 1,000, 0.5."""
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def _draw_panel(axes: plt.Axes, cells: pd.DataFrame, statistic: str, rates: list[float]) -> None:
    """One panel: a line per variant over bandwidth, whiskers for the spread of the repeats.

    `rates` are the bandwidths (Mbit/s) of the whole figure, in rising order. They set the ticks
    and the limits of the x axis: the panels share that axis, so every panel must ask for the
    same ones, also when its own model was run at fewer bandwidths.
    """
    ends = {}
    for variant, colour in VARIANT_COLOURS.items():
        line = cells[cells["variant"] == variant].sort_values("rate_mbit")
        if line.empty:
            continue
        x, y = line["rate_mbit"], line[f"{statistic}_ms"]
        # A segment that ends in an overloaded cell is faded: that end is not a level.
        points = zip(x, y, line["overloaded"], strict=True)
        for (x0, y0, lost0), (x1, y1, lost1) in itertools.pairwise(points):
            axes.plot(
                [x0, x1],
                [y0, y1],
                color=colour,
                linewidth=LINE_WIDTH,
                alpha=FADED if lost0 or lost1 else 1.0,
                solid_capstyle="round",
                zorder=2,
            )
        axes.vlines(
            x,
            line[f"{statistic}_min_ms"],
            line[f"{statistic}_max_ms"],
            color=colour,
            linewidth=WHISKER_WIDTH,
            zorder=2,
        )
        kept, lost = ~line["overloaded"], line["overloaded"]
        axes.plot(
            x[kept],
            y[kept],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=RING_WIDTH,
            zorder=3,
        )
        axes.plot(
            x[lost],
            y[lost],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE - RING_WIDTH,
            markerfacecolor=SURFACE,
            markeredgecolor=colour,
            markeredgewidth=LINE_WIDTH,
            zorder=3,
        )
        ends[variant] = (x.iloc[-1], y.iloc[-1])

    # Name each line at its end, unless two ends are so close that the names would overlap.
    heights = sorted(y for _, y in ends.values())
    apart = all(b / a >= LABEL_MIN_RATIO for a, b in zip(heights, heights[1:], strict=False))
    if apart:
        for variant, (x, y) in ends.items():
            axes.annotate(
                variant,
                (x, y),
                xytext=(LABEL_OFFSET, 0),
                textcoords="offset points",
                va="center",
                fontsize=LABEL_SIZE,
                color=INK_SECONDARY,
            )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xticks(rates)
    axes.set_xlim(rates[0] / 1.25, rates[-1] * LABEL_ROOM)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes.xaxis.set_minor_locator(NullLocator())
    axes.yaxis.set_major_formatter(FuncFormatter(_plain))
    axes.yaxis.set_minor_locator(NullLocator())
    axes.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=INK_MUTED, labelsize=LABEL_SIZE, length=3)


def plot_baselines(summary: pd.DataFrame, path: Path) -> None:
    """Draw latency against bandwidth, one line per variant, and save the figure to `path`.

    One column of panels per model; the upper row shows the median, the lower row the 95th
    percentile. Both axes are logarithmic and all panels share them. A whisker spans the lowest
    and highest repeat. Hollow markers are cells flagged as overloaded.
    """
    models = [model for model in MODEL_TITLES if model in set(summary["model"])]
    rates = sorted(summary["rate_mbit"].unique())
    width, height = PANEL_SIZE
    figure, grid = plt.subplots(
        len(STATISTICS),
        len(models),
        figsize=(width * len(models), height * len(STATISTICS)),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor=SURFACE,
    )
    for column, model in enumerate(models):
        cells = summary[summary["model"] == model]
        rate_hz = cells["rate_hz"].iloc[0]
        title = MODEL_TITLES[model]
        if pd.notna(rate_hz):
            title += f", {rate_hz:g} requests per second"
        for row, (statistic, label) in enumerate(STATISTICS):
            axes = grid[row][column]
            _draw_panel(axes, cells, statistic, rates)
            if row == 0:
                axes.set_title(title, fontsize=10, color=INK, loc="left")
            if row == len(STATISTICS) - 1:
                axes.set_xlabel("Bandwidth (Mbit/s)", fontsize=9, color=INK_SECONDARY)
            if column == 0:
                axes.set_ylabel(label, fontsize=9, color=INK_SECONDARY)

    present = [variant for variant in VARIANT_COLOURS if variant in set(summary["variant"])]
    handles = [
        Line2D(
            [],
            [],
            color=VARIANT_COLOURS[variant],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor=SURFACE,
            markeredgewidth=RING_WIDTH,
            label=variant,
        )
        for variant in present
    ]
    if summary["overloaded"].any():
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=MARKER_SIZE - RING_WIDTH,
                markerfacecolor=SURFACE,
                markeredgecolor=INK_MUTED,
                markeredgewidth=LINE_WIDTH,
                label=OVERLOADED_NOTE,
            )
        )
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=len(handles),
        frameon=False,
        fontsize=LABEL_SIZE,
        labelcolor=INK_SECONDARY,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("summary", "baselines"))
    parser.add_argument("results", type=Path, nargs="+", help="folders under results/")
    args = parser.parse_args()

    requests = pd.concat([load_runs(folder) for folder in args.results], ignore_index=True)
    summary = summarise(requests)
    table = summary.round(SUMMARY_DECIMALS)
    print(table.to_string(index=False))
    if args.what == "baselines":
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(SUMMARY_PATH, index=False)
        for suffix in (".png", ".pdf"):
            plot_baselines(summary, FIGURE_PATH.with_suffix(suffix))
        print(f"wrote {SUMMARY_PATH} and {FIGURE_PATH}.png, .pdf")


if __name__ == "__main__":
    main()
