"""scripts/train_exits.py on a tiny fake dataset: heads learn, the backbone does not move."""

import csv
import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

from edgeinf.data.imagenette import DIRNAME, WNIDS
from edgeinf.models.multi_exit import add_exits, build_model, load_checkpoint

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_exits.py"
spec = importlib.util.spec_from_file_location("train_exits", SCRIPT)
train_exits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_exits)

SEED = 0
TRAIN_PER_CLASS = 4
TEST_PER_CLASS = 1
IMAGE_SIDE = 240
EPOCHS = 2
EXITS = [2, 4]
POOL = 7
HIDDEN = 16


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
    torch.manual_seed(SEED)
    backbone = root / "backbone.pt"  # the layout scripts/train_backbone.py writes
    model = build_model("resnet18")
    torch.save({"model": model.name, "state_dict": model.state_dict(), "epochs": 1}, backbone)
    return {
        "seed": SEED,
        "backbone": str(backbone),
        "exits": EXITS,
        "head_pool": POOL,
        "head_hidden": HIDDEN,
        "data_root": str(root),
        "val_fraction": 0.25,
        "epochs": EPOCHS,
        "batch_size": 10,
        "lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.00005,
        "workers": 0,
        "device": "cpu",
        "amp": False,
    }


@pytest.fixture(scope="module")
def outputs(config, tmp_path_factory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("out")
    checkpoint, log = out / "tiny.pt", out / "tiny.csv"
    train_exits.train(config, checkpoint, log)
    return checkpoint, log


def test_checkpoint_loads_with_its_exit_heads(outputs):
    checkpoint = torch.load(outputs[0], weights_only=True)
    assert checkpoint["exits"] == {"blocks": EXITS, "pool": POOL, "hidden": HIDDEN}
    assert len(checkpoint["exit_val_accuracy"]) == len(EXITS)
    model = load_checkpoint(outputs[0])
    assert model.exit_points == EXITS
    assert not model.training


def test_backbone_is_bit_for_bit_the_one_it_started_from(outputs, config):
    backbone = torch.load(config["backbone"], weights_only=True)["state_dict"]
    trained = torch.load(outputs[0], weights_only=True)["state_dict"]
    for key, value in backbone.items():  # weights and BatchNorm statistics alike
        assert torch.equal(trained[key], value), key
    assert {key for key in trained if key not in backbone} and all(
        key.startswith("exits.") for key in trained if key not in backbone
    )


def test_training_changes_the_heads(outputs, config):
    torch.manual_seed(config["seed"])
    fresh = load_checkpoint(config["backbone"])
    add_exits(fresh, EXITS, POOL, HIDDEN)
    trained = torch.load(outputs[0], weights_only=True)["state_dict"]
    for block in EXITS:
        key = f"exits.{block}.1.weight"
        assert not torch.equal(fresh.state_dict()[key], trained[key])


def test_log_has_one_row_per_epoch_with_every_exit(outputs):
    rows = list(csv.DictReader(outputs[1].open()))
    assert [int(r["epoch"]) for r in rows] == list(range(1, EPOCHS + 1))
    assert list(rows[0]) == train_exits.log_fields(EXITS)
    for block in EXITS:
        assert all(0.0 <= float(r[f"val_accuracy_exit{block}"]) <= 1.0 for r in rows)
    # The final output is the frozen backbone's, so its accuracy cannot move between epochs.
    assert len({r["val_accuracy_final"] for r in rows}) == 1
    assert float(rows[-1]["lr"]) == pytest.approx(0.0, abs=1e-6)


def test_a_backbone_that_already_has_exits_is_refused(outputs, config, tmp_path):
    with pytest.raises(ValueError, match="already has exits"):
        train_exits.train(
            {**config, "backbone": str(outputs[0])}, tmp_path / "x.pt", tmp_path / "x.csv"
        )


def test_unknown_config_keys_are_rejected(config, tmp_path):
    with pytest.raises(ValueError, match="label_smoothing"):
        train_exits.train({**config, "label_smoothing": 0.1}, tmp_path / "x.pt", tmp_path / "x.csv")
