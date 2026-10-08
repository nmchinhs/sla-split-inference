"""The pure parts of scripts/run_experiment.py: config checks, traces, and the order of a run."""

import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
run_experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment)

VARIANTS = ["local", "offload", "offload_jpeg"]
N_FRAMES = 3925  # the test split
CONFIG = {
    "seed": 0,
    "model": "mobilenetv2",
    "checkpoint": "checkpoints/train_mobilenetv2.pt",
    "frames": "data/frames",
    "split": "test",
    "devices": ["device-1"],
    "variants": VARIANTS,
    "jpeg_quality": 75,
    "rate_hz": 5,
    "traces": ["static_1mbps", "static_50mbps"],
    "repeats": 3,
    "rounds": 4,
    "block_requests": 25,
    "block_warmup": 5,
    "warmup_requests": 50,
    "rpc_timeout_s": 5.0,
}
SHIPPED_CONFIGS = sorted(
    path
    for prefix in ("baselines", "smoke", "calibration_runtime")
    for path in (REPO_ROOT / "experiments" / "configs").glob(f"{prefix}_*.yaml")
)


# ------------------------------------------------------------------------------ config


def test_a_complete_config_is_accepted():
    run_experiment.check_config(CONFIG)
    run_experiment.check_config({**CONFIG, "rate_hz": None})  # back to back, for calibration


def test_unknown_and_missing_keys_are_refused():
    with pytest.raises(ValueError, match="bandwidth"):
        run_experiment.check_config({**CONFIG, "bandwidth": 5})
    without = {key: value for key, value in CONFIG.items() if key != "rate_hz"}
    with pytest.raises(ValueError, match="rate_hz"):
        run_experiment.check_config(without)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"variants": ["local", "joint"]}, "joint"),  # needs the scheduler of Phase 4
        ({"variants": []}, "variants"),
        ({"variants": ["local", "local"]}, "variants"),
        ({"jpeg_quality": None}, "jpeg_quality"),  # offload_jpeg is among the variants
        ({"devices": ["device-1", "device-2"]}, "one device"),  # several come with Phase 5
        ({"split": "train"}, "split"),
        ({"rate_hz": 0}, "rate_hz"),
        ({"repeats": 0}, "repeats"),
        ({"rounds": 0}, "rounds"),
        ({"block_requests": 0}, "block_requests"),
        ({"block_warmup": -1}, "block_warmup"),
        ({"traces": []}, "traces"),
        ({"model": "vgg16"}, "vgg16"),
    ],
)
def test_values_that_cannot_be_run_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        run_experiment.check_config({**CONFIG, **change})


def test_jpeg_quality_may_be_left_out_when_no_variant_uses_it():
    run_experiment.check_config({**CONFIG, "variants": ["local", "offload"], "jpeg_quality": None})


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_the_configs_in_the_repository_are_valid_and_their_traces_exist(path):
    config = run_experiment.load_config(path)
    for trace in config["traces"]:
        rate_mbit, delay_ms, loss_pct = run_experiment.static_setting(
            run_experiment.read_trace(run_experiment.TRACE_DIR / f"{trace}.csv")
        )
        assert rate_mbit > 0 and delay_ms >= 0 and loss_pct >= 0
    assert (REPO_ROOT / config["checkpoint"]).name == f"train_{config['model']}.pt"


def test_both_models_have_a_baseline_config_at_the_agreed_rates_and_bandwidths():
    rates = {}
    for path in SHIPPED_CONFIGS:
        if path.name.startswith("baselines_"):
            config = run_experiment.load_config(path)
            rates[config["model"]] = config["rate_hz"]
            assert config["variants"] == VARIANTS
            assert config["repeats"] >= 3 and config["warmup_requests"] >= 50
            assert config["traces"] == [f"static_{mbps}mbps" for mbps in (1, 5, 10, 20, 50)]
            assert config["split"] == "test"
    assert rates == {"mobilenetv2": 5, "resnet18": 2}  # report log, D19


# ------------------------------------------------------------------------------ traces


def _trace(tmp_path, text: str) -> Path:
    path = tmp_path / "trace.csv"
    path.write_text(text)
    return path


def test_a_static_trace_is_one_netem_setting(tmp_path):
    rows = run_experiment.read_trace(
        _trace(tmp_path, "time_s,rate_mbit,delay_ms,loss_pct\n0,5,20,0\n")
    )
    assert rows == [{"time_s": 0.0, "rate_mbit": 5.0, "delay_ms": 20.0, "loss_pct": 0.0}]
    assert run_experiment.static_setting(rows) == (5.0, 20.0, 0.0)


def test_a_trace_that_changes_over_time_is_refused_for_now(tmp_path):
    rows = run_experiment.read_trace(
        _trace(tmp_path, "time_s,rate_mbit,delay_ms,loss_pct\n0,50,20,0\n30,2,20,0\n")
    )
    assert len(rows) == 2
    with pytest.raises(ValueError, match="Phase 4"):
        run_experiment.static_setting(rows)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "time_s,rate_mbit,delay_ms,loss_pct\n",  # no setting at all
        "time_s,rate_mbit,delay_ms\n0,5,20\n",  # a column is missing
        "time_s,rate_mbit,delay_ms,loss_pct\n10,5,20,0\n",  # does not start at time 0
        "time_s,rate_mbit,delay_ms,loss_pct\n0,5,20,0\n0,2,20,0\n",  # time does not advance
        "time_s,rate_mbit,delay_ms,loss_pct\n0,0,20,0\n",  # no bandwidth
    ],
)
def test_malformed_traces_are_refused(tmp_path, text):
    with pytest.raises(ValueError):
        run_experiment.read_trace(_trace(tmp_path, text))


# ------------------------------------------------------------------------------ order of a run


def test_every_round_serves_every_variant_once_in_a_rotating_order():
    assert run_experiment.block_order(VARIANTS, rounds=4, shift=0) == [
        ["local", "offload", "offload_jpeg"],
        ["offload", "offload_jpeg", "local"],
        ["offload_jpeg", "local", "offload"],
        ["local", "offload", "offload_jpeg"],
    ]
    # Another repeat starts the rotation elsewhere.
    assert run_experiment.block_order(VARIANTS, rounds=1, shift=2) == [
        ["offload_jpeg", "local", "offload"]
    ]


def test_over_as_many_rounds_as_variants_each_variant_is_first_once():
    order = run_experiment.block_order(VARIANTS, rounds=len(VARIANTS), shift=1)
    assert sorted(blocks[0] for blocks in order) == sorted(VARIANTS)
    assert sorted(blocks[-1] for blocks in order) == sorted(VARIANTS)


def test_frames_are_drawn_by_seed_without_repetition():
    first = run_experiment.draw_frames(N_FRAMES, 120, "0/0/static_1mbps")
    assert first == run_experiment.draw_frames(N_FRAMES, 120, "0/0/static_1mbps")
    assert first != run_experiment.draw_frames(N_FRAMES, 120, "0/1/static_1mbps")
    assert first != run_experiment.draw_frames(N_FRAMES, 120, "0/0/static_5mbps")
    assert len(set(first)) == 120 and all(0 <= index < N_FRAMES for index in first)
    with pytest.raises(ValueError, match="frames"):
        run_experiment.draw_frames(100, 120, "0/0/static_1mbps")


def test_a_run_is_a_warmup_and_then_the_rounds():
    blocks = run_experiment.plan_run(CONFIG, repeat=0, trace="static_1mbps", n_frames=N_FRAMES)
    warmup = [block for block in blocks if block["round"] is None]
    measured = [block for block in blocks if block["round"] is not None]
    assert blocks == warmup + measured  # the warm-up comes first

    # At least 50 requests before anything is logged, spread over the variants, none logged.
    assert [block["variant"] for block in warmup] == VARIANTS
    assert sum(len(block["indices"]) for block in warmup) >= CONFIG["warmup_requests"]
    assert all(block["warmup"] == len(block["indices"]) for block in warmup)
    assert len({len(block["indices"]) for block in warmup}) == 1
    assert len(warmup[0]["indices"]) == math.ceil(CONFIG["warmup_requests"] / len(VARIANTS))

    # Then rounds x variants blocks; each logs block_requests after block_warmup unlogged ones.
    assert len(measured) == CONFIG["rounds"] * len(VARIANTS)
    assert all(block["warmup"] == CONFIG["block_warmup"] for block in measured)
    assert all(len(block["indices"]) == 30 for block in measured)
    logged = Counter()
    for block in measured:
        logged[block["variant"]] += len(block["indices"]) - block["warmup"]
    assert logged == {variant: 100 for variant in VARIANTS}
    assert [block["round"] for block in measured] == [r for r in range(4) for _ in VARIANTS]


def test_within_a_round_all_variants_serve_the_same_frames_and_rounds_differ():
    blocks = run_experiment.plan_run(CONFIG, repeat=0, trace="static_1mbps", n_frames=N_FRAMES)
    by_round: dict[int, list[list[int]]] = {}
    for block in blocks:
        if block["round"] is not None:
            by_round.setdefault(block["round"], []).append(block["indices"])
    for indices in by_round.values():
        assert indices[0] == indices[1] == indices[2]
    used = [index for indices in by_round.values() for index in indices[0]]
    assert len(set(used)) == len(used) == CONFIG["rounds"] * 30


def test_the_order_and_the_frames_depend_on_the_repeat_and_the_trace():
    def run(repeat, trace):
        return run_experiment.plan_run(CONFIG, repeat=repeat, trace=trace, n_frames=N_FRAMES)

    assert run(0, "static_1mbps") == run(0, "static_1mbps")
    first = [block["variant"] for block in run(0, "static_1mbps") if block["round"] == 0]
    second = [block["variant"] for block in run(1, "static_1mbps") if block["round"] == 0]
    assert first == VARIANTS and second == VARIANTS[1:] + VARIANTS[:1]
    assert run(0, "static_1mbps")[-1]["indices"] != run(1, "static_1mbps")[-1]["indices"]
    assert run(0, "static_1mbps")[-1]["indices"] != run(0, "static_50mbps")[-1]["indices"]


def test_a_job_tells_the_agent_everything_it_needs():
    block = {"round": 2, "variant": "offload_jpeg", "indices": [7, 3, 9], "warmup": 1}
    assert run_experiment.job(CONFIG, block) == {
        "checkpoint": "checkpoints/train_mobilenetv2.pt",
        "frames_dir": "data/frames",
        "split": "test",
        "indices": [7, 3, 9],
        "variant": "offload_jpeg",
        "jpeg_quality": 75,
        "rate_hz": 5,
        "warmup": 1,
        "timeout_s": 5.0,
    }


# ------------------------------------------------------------------------------ stale images


def _tree(root: Path) -> Path:
    (root / "edge").mkdir(parents=True)
    (root / "rpc").mkdir()
    (root / "edge" / "server.py").write_text("PORT = 50051\n")
    (root / "rpc" / "inference_pb2.py").write_text("# generated\n")
    return root


def test_the_source_hash_changes_with_the_code_and_only_with_the_code(tmp_path):
    root = _tree(tmp_path / "edgeinf")
    before = run_experiment.source_hash([], root)
    assert before == run_experiment.source_hash([], root)

    # Generated stubs differ between images, and caches are not code.
    (root / "rpc" / "inference_pb2.py").write_text("# generated by another grpcio\n")
    (root / "edge" / "__pycache__").mkdir()
    (root / "edge" / "__pycache__" / "server.cpython-311.pyc").write_bytes(b"\x00")
    assert run_experiment.source_hash([], root) == before

    (root / "edge" / "server.py").write_text("PORT = 50052\n")
    changed = run_experiment.source_hash([], root)
    assert changed != before
    (root / "edge" / "server.py").rename(root / "edge" / "service.py")
    assert run_experiment.source_hash([], root) not in (before, changed)  # a rename counts
