"""Exit profile (profiler/exit_profile.py): hand-made confidences, then a real model."""

import numpy as np
import pytest
import torch

from edgeinf.models.multi_exit import add_exits, build_model
from edgeinf.models.split import run_head_exits
from edgeinf.profiler.exit_profile import TAUS, exit_outputs, exit_profile
from edgeinf.profiler.layer_profile import PIXEL_VALUES

SEED = 0
EXIT_BLOCKS = [2, 5]
# Six inputs, two exits. Label 1 is right everywhere, so "predicted 1" means correct.
LABELS = np.ones(6, dtype=np.int64)
CONFIDENCE = np.array(
    [[0.95, 0.99], [0.95, 0.99], [0.70, 0.90], [0.55, 0.90], [0.40, 0.60], [0.40, 0.45]]
)
PREDICTED = np.array([[1, 1], [0, 1], [1, 1], [0, 1], [0, 0], [0, 1]])
FINAL = np.array([1, 1, 1, 1, 1, 0])


@pytest.fixture(scope="module")
def profile() -> dict:
    return exit_profile(CONFIDENCE, PREDICTED, FINAL, LABELS, EXIT_BLOCKS, TAUS)


def row(profile: dict, tau: float) -> dict:
    return next(r for r in profile["by_tau"] if r["tau"] == tau)


def test_profile_has_one_row_per_tau_and_the_plain_accuracies(profile):
    assert [r["tau"] for r in profile["by_tau"]] == list(TAUS)
    assert profile["exit_blocks"] == EXIT_BLOCKS
    assert profile["samples"] == 6
    assert profile["final_accuracy"] == pytest.approx(5 / 6)
    assert profile["head_accuracy"] == pytest.approx([2 / 6, 5 / 6])


def test_rates_and_accuracies_at_one_threshold_by_hand(profile):
    r = row(profile, 0.5)
    # Exit 0 takes inputs 0-3 (two right), exit 1 takes input 4 of the other two (wrong).
    assert r["exit_rate"] == pytest.approx([4 / 6, 1 / 6])
    assert r["exit_accuracy"] == pytest.approx([0.5, 0.0])
    assert r["p_continue"] == pytest.approx([2 / 6, 1 / 6])
    # Exit 0 alone: 2 right + final right on input 4, wrong on 5. Both: 2 + 0 + final wrong on 5.
    assert r["accuracy"] == pytest.approx([3 / 6, 2 / 6])


def test_an_exit_takes_only_what_earlier_exits_let_through(profile):
    r = row(profile, 0.9)
    # Exit 1 is confident on inputs 0-3, but exit 0 has already answered 0 and 1.
    assert r["exit_rate"] == pytest.approx([2 / 6, 2 / 6])
    assert r["exit_accuracy"] == pytest.approx([0.5, 1.0])
    assert r["accuracy"] == pytest.approx([4 / 6, 4 / 6])


def test_tau_one_takes_no_exit_even_at_full_confidence():
    confidence = np.ones_like(CONFIDENCE)
    r = row(exit_profile(confidence, PREDICTED, FINAL, LABELS, EXIT_BLOCKS, TAUS), 1.0)
    assert r["exit_rate"] == [0.0, 0.0]
    assert r["exit_accuracy"] == [None, None]
    assert r["p_continue"] == [1.0, 1.0]
    assert r["accuracy"] == pytest.approx([5 / 6, 5 / 6])


def test_a_lower_threshold_never_lets_more_inputs_through(profile):
    for j in range(len(EXIT_BLOCKS)):
        passing = [r["p_continue"][j] for r in profile["by_tau"]]  # taus ascend
        assert passing == sorted(passing)
    for r in profile["by_tau"]:
        assert r["p_continue"][-1] == pytest.approx(1.0 - sum(r["exit_rate"]))


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        exit_profile(CONFIDENCE, PREDICTED, FINAL, LABELS, [2], TAUS)
    with pytest.raises(ValueError):
        exit_profile(CONFIDENCE, PREDICTED, FINAL[:3], LABELS, EXIT_BLOCKS, TAUS)


def test_profile_agrees_with_the_device_walk_on_a_real_model(settle_batchnorm):
    torch.manual_seed(SEED)
    model = build_model("resnet18")
    add_exits(model, [1, 3], pool=7, hidden=16)
    settle_batchnorm(model)
    model.eval()
    generator = torch.Generator().manual_seed(SEED)
    frames = torch.randint(0, PIXEL_VALUES, (5, 3, 224, 224), generator=generator)
    frames = frames.to(torch.uint8).numpy()
    labels = np.zeros(len(frames), dtype=np.int64)

    confidence, predicted, final = exit_outputs(model, frames, batch_size=2)
    assert confidence.shape == predicted.shape == (5, 2) and final.shape == (5,)
    # Between two confidences of the first exit: it takes three of the five frames, and none
    # sits so close to tau that the rounding of a batch could flip its decision.
    first = np.sort(confidence[:, 0])
    tau = float((first[1] + first[2]) / 2)
    profile = exit_profile(confidence, predicted, final, labels, model.exit_points, [tau])
    row = profile["by_tau"][0]
    assert row["exit_rate"][0] == pytest.approx(3 / 5)

    n = model.num_blocks
    taken_at = []
    with torch.inference_mode():
        for frame in torch.from_numpy(frames):
            _, taken = run_head_exits(model, frame[None], n, tau)
            taken_at.append(None if taken is None else taken.block)
    for j, block in enumerate(model.exit_points):
        assert row["exit_rate"][j] == pytest.approx(taken_at.count(block) / len(frames))
    assert row["p_continue"][-1] == pytest.approx(taken_at.count(None) / len(frames))
