"""Calibrate the CPU quota of the device containers (PLAN.md Phase 2 step 5).

A device container is slowed down with a CFS quota: `quota_us` of CPU time in every `period_us`.
This script measures how long the device part of a request takes under a list of (quota, period)
settings, for requests that arrive back to back ("burst") and for requests separated by random
idle gaps ("spaced"). The aim is a setting that matches a published Raspberry Pi latency and
does not depend on how requests are spaced.

Run on the host, with the device image built:

    python scripts/calibrate_cpu.py experiments/configs/<name>.yaml

It starts one throwaway container from the device image, pinned to one CPU, with this working
tree's src/ and scripts/ mounted read-only, and changes that container's quota with
`docker update` between measurements. Inside the container this same file runs as a probe
(`--probe`) that keeps the models loaded and times requests on demand.

Outputs: one row per request in results/calibration/<name>.csv, and one row per
(setting, model, k, pattern) in experiments/calibration/<name>.csv.

With `--cpu N` the container is pinned to host CPU N instead of the config's `cpu`, and the
output files are named <name>_cpu<N>.csv. Two such runs started together measure how much two
devices disturb each other, for example on the two hardware threads of one core.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "results" / "calibration"
SUMMARY_DIR = REPO_ROOT / "experiments" / "calibration"
# The host CPU is appended, so sweeps on different CPUs can run side by side.
CONTAINER_PREFIX = "edgeinf-calibration-cpu"
CGROUP_CPU_STAT = Path("/sys/fs/cgroup/cpu.stat")
CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")  # "<quota_us> <period_us>" as the kernel has it

# Limits that Docker enforces on --cpu-period and --cpu-quota, in microseconds.
MIN_PERIOD_US = 1_000
MAX_PERIOD_US = 1_000_000
MIN_QUOTA_US = 1_000

P95 = 0.95
MS_PER_S = 1e3
NS_PER_MS = 1e6
US_PER_MS = 1e3

RAW_FIELDS = (
    "quota_us",
    "period_us",
    "repeat",
    "model",
    "k",
    "pattern",
    "latency_ms",
    "cpu_ms",
    "page_faults",
)
SUMMARY_FIELDS = (
    "quota_us",
    "period_us",
    "share",
    "model",
    "k",
    "pattern",
    "repeats",
    "median_ms",
    "median_min_ms",
    "median_max_ms",
    "p95_ms",
    "cpu_ms",
    "page_faults",
    "achieved_share",
)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value with at least `fraction` of the data <= it."""
    ordered = sorted(values)
    return ordered[max(math.ceil(fraction * len(ordered)), 1) - 1]


def check_setting(quota_us: int, period_us: int) -> None:
    """Raise ValueError unless Docker accepts this quota and period for a one-CPU container.

    Accepted is not the same as enforced: with the systemd cgroup driver Docker rounds the share
    quota / period up to a whole percent of a CPU. `sweep` warns when that changes a setting.
    """
    if not MIN_PERIOD_US <= period_us <= MAX_PERIOD_US:
        raise ValueError(f"period {period_us} us is outside [{MIN_PERIOD_US}, {MAX_PERIOD_US}]")
    if not MIN_QUOTA_US <= quota_us <= period_us:
        raise ValueError(f"quota {quota_us} us is outside [{MIN_QUOTA_US}, {period_us}]")


def summarise(rows: list[dict], shares: dict[tuple, list[float]]) -> list[dict]:
    """One row per (quota, period, model, k, pattern), from the per-request `rows`.

    The median and p95 latency are computed per repeat and then averaged over the repeats;
    `median_min_ms` and `median_max_ms` give the spread of the per-repeat medians. `shares` maps
    the same key to the fraction of wall time the container was on the CPU in each repeat.
    """
    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["quota_us"], row["period_us"], row["model"], row["k"], row["pattern"])
        groups[key][row["repeat"]].append(row)

    summary = []
    for key, by_repeat in groups.items():
        quota_us, period_us, model, k, pattern = key
        medians = [
            statistics.median(r["latency_ms"] for r in repeat) for repeat in by_repeat.values()
        ]
        p95s = [percentile([r["latency_ms"] for r in rep], P95) for rep in by_repeat.values()]
        cpu = [r["cpu_ms"] for repeat in by_repeat.values() for r in repeat]
        faults = [r["page_faults"] for repeat in by_repeat.values() for r in repeat]
        share = shares.get(key)  # only measured for the burst pattern
        summary.append(
            {
                "quota_us": quota_us,
                "period_us": period_us,
                "share": round(quota_us / period_us, 4),
                "model": model,
                "k": k,
                "pattern": pattern,
                "repeats": len(medians),
                "median_ms": round(statistics.mean(medians), 2),
                "median_min_ms": round(min(medians), 2),
                "median_max_ms": round(max(medians), 2),
                "p95_ms": round(statistics.mean(p95s), 2),
                "cpu_ms": round(statistics.mean(cpu), 2),
                "page_faults": round(statistics.mean(faults), 1),
                "achieved_share": round(statistics.mean(share), 4) if share else "",
            }
        )
    return summary


# ---------------------------------------------------------------- probe (inside the container)


def _cpu_stat() -> dict[str, int]:
    """The container's cgroup CPU counters (usage_usec, nr_periods, nr_throttled, ...)."""
    try:
        lines = CGROUP_CPU_STAT.read_text().splitlines()
    except OSError:
        return {}
    return {name: int(value) for name, value in (line.split() for line in lines)}


def probe() -> None:
    """Serve timing requests from stdin, one JSON object per line, until stdin closes.

    Request: {"model", "k", "requests", "warmup", "gap_ms": [low, high] or null, "seed"}.
    Reply: per-request wall time (ms), CPU time (ms) and page faults for running blocks [0, k) on
    one image, and the share of wall time the container spent on the CPU while they were timed.

    Weights are random: the time does not depend on their values. Nothing larger than one
    request is ever allocated here, because freeing a large block changes how glibc's malloc
    behaves for the rest of the process (see experiments/calibration.md).
    """
    # Imported here so that the host side of this script does not need torch.
    import torch

    from edgeinf.models.multi_exit import INPUT_SIZE, build_model
    from edgeinf.models.split import run_blocks

    torch.set_num_threads(1)  # a device has one core
    models = {}
    for line in sys.stdin:
        request = json.loads(line)
        torch.manual_seed(request["seed"])
        name = request["model"]
        if name not in models:
            models[name] = build_model(name).eval()
        model = models[name]
        image = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
        rng = random.Random(request["seed"])
        gap_ms = request["gap_ms"]

        latency_ms, cpu_ms, page_faults = [], [], []
        with torch.inference_mode():
            for i in range(request["warmup"] + request["requests"]):
                if i == request["warmup"]:
                    stat_start, wall_start = _cpu_stat(), time.perf_counter_ns()
                if gap_ms is not None:
                    time.sleep(rng.uniform(*gap_ms) / MS_PER_S)
                faults_start = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
                cpu_start, start = time.thread_time_ns(), time.perf_counter_ns()
                run_blocks(model, image, 0, request["k"])
                elapsed_ns = time.perf_counter_ns() - start
                cpu_ns = time.thread_time_ns() - cpu_start
                faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults_start
                if i >= request["warmup"]:
                    latency_ms.append(elapsed_ns / NS_PER_MS)
                    cpu_ms.append(cpu_ns / NS_PER_MS)
                    page_faults.append(faults)
        stat_end, wall_ms = _cpu_stat(), (time.perf_counter_ns() - wall_start) / NS_PER_MS
        usage_ms = (stat_end.get("usage_usec", 0) - stat_start.get("usage_usec", 0)) / US_PER_MS
        reply = {
            "latency_ms": latency_ms,
            "cpu_ms": cpu_ms,
            "page_faults": page_faults,
            # Only meaningful for the burst pattern, where the container is never idle by choice.
            "achieved_share": usage_ms / wall_ms if stat_end else None,
        }
        print(json.dumps(reply), flush=True)


# ---------------------------------------------------------------- sweep (on the host)


def _docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def sweep(config_path: Path, cpu: int | None = None) -> None:
    """Run the sweep described by the YAML file at `config_path` and write both CSV files.

    `cpu` overrides the host CPU named in the config and is added to the output file names.
    """
    config = yaml.safe_load(config_path.read_text())
    name = config_path.stem
    if cpu is not None:
        config["cpu"] = cpu
        name = f"{name}_cpu{cpu}"
    settings = [tuple(setting) for setting in config["settings"]]
    for quota_us, period_us in settings:
        check_setting(quota_us, period_us)
    patterns = {"burst": None, "spaced": config["gap_ms"]}
    container = f"{CONTAINER_PREFIX}{config['cpu']}"

    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    _docker(
        "run",
        "-d",
        "--rm",
        f"--name={container}",
        f"--cpuset-cpus={config['cpu']}",
        f"--memory={config['memory']}",
        f"--cpu-period={settings[0][1]}",
        f"--cpu-quota={settings[0][0]}",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        *(f"--env={name}={value}" for name, value in (config.get("env") or {}).items()),
        "-v",
        f"{REPO_ROOT / 'src'}:/app/src:ro",
        "-v",
        f"{REPO_ROOT / 'scripts'}:/app/scripts:ro",
        config["image"],
    )
    rows: list[dict] = []
    shares: dict[tuple, list[float]] = defaultdict(list)
    try:
        prober = subprocess.Popen(
            ["docker", "exec", "-i", container, "python", "scripts/calibrate_cpu.py", "--probe"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        for repeat in range(config["repeats"]):
            order = settings.copy()
            random.Random(config["seed"] + repeat).shuffle(order)
            for quota_us, period_us in order:
                _docker("update", f"--cpu-period={period_us}", f"--cpu-quota={quota_us}", container)
                enforced = _docker("exec", container, "cat", str(CGROUP_CPU_MAX))
                if repeat == 0 and enforced != f"{quota_us} {period_us}":
                    print(
                        f"warning: asked for {quota_us} us per {period_us} us, "
                        f"the kernel enforces {enforced} (cpu.max)",
                        file=sys.stderr,
                        flush=True,
                    )
                for model, split_points in config["models"].items():
                    for k in split_points:
                        for pattern, gap_ms in patterns.items():
                            request = {
                                "model": model,
                                "k": k,
                                "requests": config["requests"],
                                "warmup": config["warmup"],
                                "gap_ms": gap_ms,
                                "seed": config["seed"] + repeat,
                            }
                            prober.stdin.write(json.dumps(request) + "\n")
                            prober.stdin.flush()
                            line = prober.stdout.readline()
                            if not line:
                                raise RuntimeError("the probe in the container stopped")
                            reply = json.loads(line)
                            key = (quota_us, period_us, model, k, pattern)
                            if pattern == "burst" and reply["achieved_share"] is not None:
                                shares[key].append(reply["achieved_share"])
                            for latency, cpu, faults in zip(
                                reply["latency_ms"],
                                reply["cpu_ms"],
                                reply["page_faults"],
                                strict=True,
                            ):
                                rows.append(
                                    {
                                        "quota_us": quota_us,
                                        "period_us": period_us,
                                        "repeat": repeat,
                                        "model": model,
                                        "k": k,
                                        "pattern": pattern,
                                        "latency_ms": round(latency, 3),
                                        "cpu_ms": round(cpu, 3),
                                        "page_faults": faults,
                                    }
                                )
                print(
                    f"repeat {repeat}: quota {quota_us} us / period {period_us} us done",
                    file=sys.stderr,
                    flush=True,
                )
        prober.stdin.close()
        prober.wait()
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)

    summary = summarise(rows, shares)
    summary.sort(key=lambda r: (r["model"], r["k"], r["share"], r["quota_us"], r["pattern"]))
    for directory, fields, table in (
        (RAW_DIR, RAW_FIELDS, rows),
        (SUMMARY_DIR, SUMMARY_FIELDS, summary),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    print(f"wrote {RAW_DIR / name}.csv ({len(rows)} requests)")
    print(f"wrote {SUMMARY_DIR / name}.csv ({len(summary)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, nargs="?", help="YAML file describing the sweep")
    parser.add_argument("--probe", action="store_true", help="run as the in-container probe")
    parser.add_argument("--cpu", type=int, help="host CPU to pin to, instead of the config's")
    args = parser.parse_args()
    if args.probe:
        probe()
    elif args.config is None:
        parser.error("give a config file, or --probe inside the container")
    else:
        sweep(args.config, args.cpu)


if __name__ == "__main__":
    main()
