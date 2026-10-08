"""scripts/train_backbone.py on a tiny fake dataset: one epoch on the CPU, checkpoint and log."""

import csv
import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

from edgeinf.data.imagenette import DIRNAME, WNIDS
from edgeinf.models.multi_exit import build_model

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_backbone.py"
spec = importlib.util.spec_from_file_location("train_backbone", SCRIPT)
train_backbone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_backbone)

SEED = 0
TRAIN_PER_CLASS = 4
TEST_PER_CLASS = 1
IMAGE_SIDE = 240
EPOCHS = 2
LR = 0.05


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("data")
    generator = torch.Generator().manual_seed(SEED)
    for split, per_class in (("train", TRAIN_PER_CLASS), ("val", TEST_PER_CLASS)):
        for wnid in WNIDS:
            folder = root / DIRNAME / split / wnid
            folder.mkdir(parents=True)
            for i in range(per_class):
                pixels = torch.randint(0, 256, (IMAGE_SIDE, IMAGE_SIDE, 3), generator=generator)
                Image.fromarray(pixels.to(torch.uint8).numpy()).save(folder / f"{i}.JPEG")
    return {
        "seed": SEED,
        "model": "mobilenetv2",
        "data_root": str(root),
        "val_fraction": 0.25,
        "epochs": EPOCHS,
        "batch_size": 10,
        "lr": LR,
        "momentum": 0.9,
        "weight_decay": 0.00004,
        "label_smoothing": 0.1,
        "warmup_epochs": 1,
        "workers": 0,
        "device": "cpu",
        "amp": False,
    }


@pytest.fixture(scope="module")
def outputs(config, tmp_path_factory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("out")
    checkpoint, log = out / "tiny.pt", out / "tiny.csv"
    train_backbone.train(config, checkpoint, log)
    return checkpoint, log


def test_checkpoint_loads_into_a_freshly_built_model(outputs, config):
    checkpoint = torch.load(outputs[0], weights_only=True)
    model = build_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])  # strict: exactly the backbone's keys
    assert checkpoint["model"] == config["model"]
    assert checkpoint["epochs"] == EPOCHS
    assert checkpoint["config"]["seed"] == SEED
    assert 0.0 <= checkpoint["val_accuracy"] <= 1.0


def test_training_changes_the_weights(outputs, config):
    torch.manual_seed(config["seed"])
    initial = build_model(config["model"]).state_dict()
    trained = torch.load(outputs[0], weights_only=True)["state_dict"]
    assert not torch.equal(initial["blocks.0.0.weight"], trained["blocks.0.0.weight"])


def test_log_has_one_row_per_epoch(outputs):
    rows = list(csv.DictReader(outputs[1].open()))
    assert [int(r["epoch"]) for r in rows] == list(range(1, EPOCHS + 1))
    assert set(train_backbone.LOG_FIELDS) == set(rows[0])
    assert all(0.0 <= float(r["val_accuracy"]) <= 1.0 for r in rows)
    # The schedule is really applied: the rate is up after the warm-up epoch and zero at the end.
    assert float(rows[0]["lr"]) > 0.5 * LR
    assert float(rows[-1]["lr"]) == pytest.approx(0.0, abs=1e-6)


def test_learning_rate_warms_up_then_decays_to_zero():
    steps_per_epoch, epochs, warmup = 10, 5, 1
    lr = [
        train_backbone.lr_factor(step, steps_per_epoch * warmup, steps_per_epoch * epochs)
        for step in range(steps_per_epoch * epochs)
    ]
    assert lr[0] == pytest.approx(1 / (steps_per_epoch * warmup))  # starts near zero
    assert lr[4] == pytest.approx(0.5)  # and rises in a straight line
    assert lr[0] < lr[5] < lr[9] <= 1.0
    assert max(lr) == pytest.approx(1.0)
    assert lr[-1] < 0.01
    assert all(a >= b for a, b in zip(lr[10:], lr[11:], strict=False))


def test_unknown_config_keys_are_rejected(config, tmp_path):
    with pytest.raises(ValueError, match="learning_rate"):
        train_backbone.train(
            {**config, "learning_rate": 0.1}, tmp_path / "x.pt", tmp_path / "x.csv"
        )
