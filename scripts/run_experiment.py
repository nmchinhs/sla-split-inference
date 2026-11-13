"""Run one experiment config end to end and write one CSV row per request to results/.

    python scripts/run_experiment.py experiments/configs/<name>.yaml [--force]

Run on the host, with the testbed up (`docker compose up -d --build`) and the machine otherwise
idle. A config names one model, the variants to compare, and a list of bandwidth traces. Every
trace is run `repeats` times; one such run writes

    results/<name>/<trace>_r<repeat>.csv    one row per request (columns: common/logging.py)
    results/<name>/<trace>_r<repeat>.json   the config, the trace, the code version, the blocks

A run whose CSV exists is skipped, so an interrupted experiment continues where it stopped;
`--force` runs everything again.

How a run goes. The link of the device is set with scripts/netem.sh. The variants are not run one
after another but in blocks that take turns: in every round each variant serves one block of
requests, and the order rotates from round to round. That way a drift of the laptop's speed over
minutes hits all variants alike. A block has its own arrival clock and starts only when the block
before it has been served completely, so a variant that cannot keep up delays its own requests
and nobody else's. Before the first round `warmup_requests` requests are served and not logged,
and neither are the first `block_warmup` requests of each block. All variants of a round serve
the same frames.

The requests themselves are served by an agent process in the device container
(src/edgeinf/device/client.py), which gets one job per block and returns its rows.

An entry of `variants` is the name of a variant, or a plan written out,

    - {variant: pinned, k: 5, tau: 0.9}

which serves every request with that split point and threshold. Pinned plans are for checking the
predictor; in the CSV their variant is `pinned`, and k and tau tell them apart.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from edgeinf.common.logging import RequestLog
from edgeinf.data.imagenette import FRAME_SPLITS, LABELS_FILE
from edgeinf.device.client import PINNED, VARIANTS
from edgeinf.models.multi_exit import MODEL_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
TRACE_DIR = REPO_ROOT / "experiments" / "traces"
NETEM = REPO_ROOT / "scripts" / "netem.sh"
SOURCE = REPO_ROOT / "src" / "edgeinf"
SOURCE_IN_CONTAINER = "/app/src/edgeinf"
EDGE = "edge"  # name of the edge container
CGROUP_CPU_MAX = "/sys/fs/cgroup/cpu.max"
# The runner keeps to the CPUs of the OS and the edge and never runs on a device's
# (scripts/gen_compose.py, RESERVED_CORES).
RUNNER_CPUS = (0, 1)
AGENT_EXIT_S = 10.0  # how long the agent gets to stop when the experiment ends

CONFIG_KEYS = {
    "seed",
    "model",
    "checkpoint",
    "frames",
    "split",
    "devices",
    "variants",
    "jpeg_quality",
    "rate_hz",
    "traces",
    "repeats",
    "rounds",
    "block_requests",
    "block_warmup",
    "warmup_requests",
    "rpc_timeout_s",
}
TRACE_FIELDS = ("time_s", "rate_mbit", "delay_ms", "loss_pct")
MAX_LOSS_PCT = 100.0

# Printed by a container or by the host: one hash over the package's source files. Generated gRPC
# stubs are left out, because every image generates its own.
HASH_CODE = """
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    relative = path.relative_to(root).as_posix()
    if not relative.startswith("rpc/"):
        digest.update(relative.encode() + b"\\0" + path.read_bytes() + b"\\0")
print(digest.hexdigest())
"""


# ------------------------------------------------------------------------------ config and traces


def check_config(config: dict) -> None:
    """Raise ValueError unless `config` is a complete experiment config that can be run."""
    if unknown := set(config) - CONFIG_KEYS:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if missing := CONFIG_KEYS - set(config):
        raise ValueError(f"missing config keys: {sorted(missing)}")
    if config["model"] not in MODEL_NAMES:
        raise ValueError(f"unknown model {config['model']!r}; expected one of {MODEL_NAMES}")
    variants = [plan["variant"] for plan in plans(config)]
    if "offload_jpeg" in variants and config["jpeg_quality"] is None:
        raise ValueError("offload_jpeg is among the variants, so jpeg_quality must be set")
    if len(config["devices"]) != 1:
        raise ValueError(f"exactly one device is run until Phase 5, got {config['devices']}")
    if config["split"] not in FRAME_SPLITS:
        raise ValueError(f"split must be one of {FRAME_SPLITS}, got {config['split']!r}")
    if config["rate_hz"] is not None and config["rate_hz"] <= 0:
        raise ValueError("rate_hz must be positive, or null for back-to-back requests")
    for key in ("repeats", "rounds", "block_requests"):
        if config[key] < 1:
            raise ValueError(f"{key} must be at least 1, got {config[key]}")
    for key in ("block_warmup", "warmup_requests"):
        if config[key] < 0:
            raise ValueError(f"{key} must not be negative, got {config[key]}")
    if not config["traces"]:
        raise ValueError("traces must name at least one trace")


def plans(config: dict) -> list[dict]:
    """The entries of `variants` as {"label", "variant"} and, for a pinned plan, "k" and "tau".

    The label names the entry in the run's record and in the messages: the variant's name, or
    "pinned k=5 tau=0.9". Raises ValueError for an entry that cannot be run, and unless all
    labels differ.
    """
    entries = []
    for entry in config["variants"] or ():
        if isinstance(entry, str):
            if entry not in VARIANTS:
                raise ValueError(
                    f"variant {entry!r} cannot be run yet; expected one of {VARIANTS} "
                    f"or a {PINNED} plan"
                )
            entries.append({"label": entry, "variant": entry})
            continue
        if not isinstance(entry, dict) or set(entry) != {"variant", "k", "tau"}:
            raise ValueError(f"a plan must have the keys variant, k and tau, got {entry!r}")
        k, tau = entry["k"], entry["tau"]
        if entry["variant"] != PINNED:
            raise ValueError(f"only a {PINNED} plan is written out, got {entry['variant']!r}")
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError(f"k of a pinned plan must be a split point, got {k!r}")
        if isinstance(tau, bool) or not isinstance(tau, int | float) or not 0 < tau <= 1:
            raise ValueError(f"tau of a pinned plan must be in (0, 1], got {tau!r}")
        entries.append(
            {"label": f"{PINNED} k={k} tau={tau}", "variant": PINNED, "k": k, "tau": tau}
        )
    labels = [entry["label"] for entry in entries]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(f"variants must be a list of different variants, got {labels}")
    return entries


def load_config(path: Path) -> dict:
    """Read and check the experiment config at `path`."""
    config = yaml.safe_load(path.read_text())
    check_config(config)
    return config


def read_trace(path: Path) -> list[dict[str, float]]:
    """The settings of a bandwidth trace: rate in Mbit/s, delay in ms and loss in % from time_s on.

    Raises ValueError unless the file has the columns of TRACE_FIELDS, starts at time 0 and
    moves forward in time.
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != TRACE_FIELDS:
            raise ValueError(f"{path} must have the columns {TRACE_FIELDS}")
        rows = [{name: float(row[name]) for name in TRACE_FIELDS} for row in reader]
    if not rows or rows[0]["time_s"] != 0:
        raise ValueError(f"{path} must start with a setting at time 0")
    if any(later["time_s"] <= earlier["time_s"] for earlier, later in itertools.pairwise(rows)):
        raise ValueError(f"{path}: time_s must increase from row to row")
    for row in rows:
        if row["rate_mbit"] <= 0 or row["delay_ms"] < 0 or not 0 <= row["loss_pct"] <= MAX_LOSS_PCT:
            raise ValueError(f"{path}: not a link setting: {row}")
    return rows


def static_setting(rows: list[dict[str, float]]) -> tuple[float, float, float]:
    """(rate in Mbit/s, delay in ms, loss in %) of a trace that never changes."""
    if len(rows) != 1:
        raise ValueError("traces that change over time are replayed from Phase 4 on")
    return rows[0]["rate_mbit"], rows[0]["delay_ms"], rows[0]["loss_pct"]


# ------------------------------------------------------------------------------ the order of a run


def block_order(variants: list[str], rounds: int, shift: int) -> list[list[str]]:
    """For each round, the order in which the variants serve their block.

    Round r starts with variant (r + shift) and goes on cyclically, so over as many rounds as
    there are variants each one is first once and last once.
    """
    count = len(variants)
    return [
        [variants[(round_ + shift + position) % count] for position in range(count)]
        for round_ in range(rounds)
    ]


def draw_frames(n_frames: int, count: int, key: str) -> list[int]:
    """`count` different frame indices below `n_frames`, fixed by `key`."""
    if count > n_frames:
        raise ValueError(f"the run needs {count} different frames, the split has {n_frames}")
    return random.Random(key).sample(range(n_frames), count)


def plan_run(config: dict, repeat: int, trace: str, n_frames: int) -> list[dict]:
    """The blocks of one run in the order they are served.

    A block is {"round", "variant", "indices", "warmup"}: the frames to serve and how many of
    the first requests are not logged; "variant" is the label of an entry of `plans`. The
    warm-up blocks come first, with "round" None and nothing logged. The frames depend on the
    seed, the repeat and the trace; all variants of a round get the same ones.
    """
    variants = [plan["label"] for plan in plans(config)]
    per_block = config["block_warmup"] + config["block_requests"]
    warmup_each = math.ceil(config["warmup_requests"] / len(variants))
    key = f"{config['seed']}/{repeat}/{trace}"
    frames = draw_frames(n_frames, max(config["rounds"] * per_block, warmup_each), key)

    blocks = []
    if warmup_each:
        blocks += [
            {
                "round": None,
                "variant": variant,
                "indices": frames[:warmup_each],
                "warmup": warmup_each,
            }
            for variant in variants
        ]
    for round_, order in enumerate(block_order(variants, config["rounds"], shift=repeat)):
        indices = frames[round_ * per_block : (round_ + 1) * per_block]
        blocks += [
            {
                "round": round_,
                "variant": variant,
                "indices": indices,
                "warmup": config["block_warmup"],
            }
            for variant in order
        ]
    return blocks


def job(config: dict, block: dict) -> dict:
    """The job that makes the device agent serve `block` (see Agent in device/client.py)."""
    plan = next(plan for plan in plans(config) if plan["label"] == block["variant"])
    return {
        "checkpoint": config["checkpoint"],
        "frames_dir": config["frames"],
        "split": config["split"],
        "indices": block["indices"],
        "jpeg_quality": config["jpeg_quality"],
        "rate_hz": config["rate_hz"],
        "warmup": block["warmup"],
        "timeout_s": config["rpc_timeout_s"],
    } | {key: value for key, value in plan.items() if key != "label"}


# ------------------------------------------------------------------------------ the testbed


def _output(*command: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command[:4])} ... failed: {result.stderr.strip()}")
    return result.stdout.strip()


def source_hash(prefix: list[str], root: str | Path) -> str:
    """Hash of the package source under `root`, computed by `prefix` + python.

    An empty prefix computes it here; ["docker", "exec", <container>] computes it in a container.
    """
    python = [*prefix, "python"] if prefix else [sys.executable]
    return _output(*python, "-c", HASH_CODE, str(root))


def preflight(config: dict) -> dict:
    """Check that the testbed can run `config`; return facts about it for the run's record."""
    device = config["devices"][0]
    for container in (EDGE, device):
        try:
            running = _output("docker", "inspect", "-f", "{{.State.Running}}", container)
        except RuntimeError:
            running = "false"
        if running != "true":
            raise RuntimeError(
                f"container {container} is not running: docker compose up -d --build"
            )
    here = source_hash([], SOURCE)
    for container in (EDGE, device):
        if source_hash(["docker", "exec", container], SOURCE_IN_CONTAINER) != here:
            raise RuntimeError(
                f"the code in container {container} is not the code in this working tree: "
                "docker compose up -d --build"
            )
    served = _output("docker", "exec", EDGE, "printenv", "EDGE_CHECKPOINTS").split(",")
    if config["checkpoint"] not in served:
        raise RuntimeError(f"the edge serves {served}, not {config['checkpoint']}")
    if not (REPO_ROOT / config["checkpoint"]).is_file():
        raise RuntimeError(f"{config['checkpoint']} does not exist")
    return {
        "source_hash": here,
        "git_commit": _output("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
        "git_dirty": bool(_output("git", "-C", str(REPO_ROOT), "status", "--porcelain")),
        "device_cpu_max": _output("docker", "exec", device, "cat", CGROUP_CPU_MAX),
        "device_glibc_tunables": _output(
            "docker", "exec", device, "sh", "-c", "echo $GLIBC_TUNABLES"
        ),
        "edge_cpus": _output("docker", "inspect", "-f", "{{.HostConfig.CpusetCpus}}", EDGE),
    }


def netem(container: str, *setting: object) -> None:
    """Set the link of `container` with scripts/netem.sh: (rate, delay, loss), or ("clear",)."""
    _output(str(NETEM), container, *(str(value) for value in setting))


class AgentProcess:
    """The device agent in `container`, started with docker exec and fed one job at a time."""

    def __init__(self, container: str) -> None:
        self.container = container
        self._process = subprocess.Popen(
            ["docker", "exec", "-i", container, "python", "-m", "edgeinf.device.client"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def run(self, job: dict) -> dict:
        """Send one job and wait for its reply."""
        self._process.stdin.write(json.dumps(job) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"the agent in {self.container} stopped; its messages are above")
        return json.loads(line)

    def close(self) -> None:
        """End the agent: it stops when its input closes."""
        self._process.stdin.close()
        try:
            self._process.wait(AGENT_EXIT_S)
        except subprocess.TimeoutExpired:
            self._process.kill()


# ------------------------------------------------------------------------------ running


def run_once(
    config: dict, agent: AgentProcess, repeat: int, trace: str, n_frames: int, csv_path: Path
) -> list[dict]:
    """Serve the blocks of one run, write its rows to `csv_path`, return a record per block."""
    records, medians, fallbacks = [], {}, 0
    with RequestLog(csv_path) as log:
        for block in plan_run(config, repeat, trace, n_frames):
            started = time.time()
            reply = agent.run(job(config, block))
            for row in reply["rows"]:
                log.write(row)
                fallbacks += not row["edge_up"] and row["k"] < reply["n_blocks"]
            records.append(
                {
                    "round": block["round"],
                    "variant": block["variant"],
                    "requests": len(block["indices"]),
                    "logged": len(reply["rows"]),
                    "started": started,
                    "seconds": round(time.time() - started, 3),
                    "minor_faults": reply["minor_faults"],
                    "n_blocks": reply["n_blocks"],
                    "link": reply["link"],  # the device's estimate when the block ended
                }
            )
            medians.setdefault(block["variant"], []).extend(
                row["latency_ms"] for row in reply["rows"]
            )
    summary = ", ".join(
        f"{variant} {statistics.median(values):.0f} ms"
        for variant, values in medians.items()
        if values
    )
    note = f"; {fallbacks} requests fell back to the device" if fallbacks else ""
    print(f"  repeat {repeat} {trace}: median latency {summary}{note}", flush=True)
    return records


def run(config_path: Path, force: bool = False) -> None:
    """Run the experiment described by the YAML file at `config_path`."""
    config = load_config(config_path)
    out_dir = RESULTS_DIR / config_path.stem
    traces = {
        name: static_setting(read_trace(TRACE_DIR / f"{name}.csv")) for name in config["traces"]
    }
    labels = REPO_ROOT / config["frames"] / LABELS_FILE.format(split=config["split"])
    if not labels.is_file():
        raise RuntimeError(f"{labels} not found: run scripts/export_frames.py first")
    n_frames = len(np.load(labels))
    os.sched_setaffinity(0, RUNNER_CPUS)
    testbed = preflight(config)
    device = config["devices"][0]

    names = [plan["label"] for plan in plans(config)]
    print(f"{config_path.stem}: {config['model']}, {names}, writing to {out_dir}")
    agent = AgentProcess(device)
    try:
        for repeat in range(config["repeats"]):
            order = list(traces)
            random.Random(config["seed"] + repeat).shuffle(order)
            for trace in order:
                csv_path = out_dir / f"{trace}_r{repeat}.csv"
                if csv_path.exists() and not force:
                    print(f"  repeat {repeat} {trace}: {csv_path.name} exists, skipped", flush=True)
                    continue
                rate_mbit, delay_ms, loss_pct = traces[trace]
                netem(device, rate_mbit, delay_ms, loss_pct)
                started = datetime.datetime.now().astimezone()
                partial = csv_path.with_name(csv_path.name + ".part")
                blocks = run_once(config, agent, repeat, trace, n_frames, partial)
                record = {
                    "config_file": str(config_path),
                    "config": config,
                    "trace": {
                        "name": trace,
                        "rate_mbit": rate_mbit,
                        "delay_ms": delay_ms,
                        "loss_pct": loss_pct,
                    },
                    "repeat": repeat,
                    "started": started.isoformat(timespec="seconds"),
                    "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "testbed": testbed,
                    "blocks": blocks,
                }
                csv_path.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
                partial.rename(csv_path)  # last: a CSV exists only for a run that is complete
    finally:
        agent.close()
        netem(device, "clear")
    print(f"done: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML file describing the experiment")
    parser.add_argument("--force", action="store_true", help="run again what already has a CSV")
    args = parser.parse_args()
    run(args.config, args.force)


if __name__ == "__main__":
    main()
