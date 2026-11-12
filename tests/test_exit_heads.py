"""Exit heads (models/multi_exit.py) and the device's exit checks (models/split.py)."""

import pytest
import torch

from edgeinf.data.imagenette import normalise
from edgeinf.models.multi_exit import (
    MODEL_NAMES,
    NUM_CLASSES,
    add_exits,
    block_channels,
    build_model,
    exit_head,
    load_checkpoint,
)
from edgeinf.models.split import (
    TAU_NEVER,
    Exit,
    exit_check,
    run_blocks,
    run_head,
    run_head_exits,
    run_head_forced,
)

SEED = 0
OTHER_SEED = 1
POOL = 7
HIDDEN = 16
EXITS = {"mobilenetv2": [4, 10, 14], "resnet18": [2, 4, 6]}
CHANNELS = {"mobilenetv2": [32, 64, 160], "resnet18": [64, 128, 256]}
TAU_ALWAYS = 0.0  # every confidence is at least this


@pytest.fixture(autouse=True)
def _inference_mode():
    with torch.inference_mode():
        yield


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request, settle_batchnorm):
    torch.manual_seed(SEED)
    model = build_model(request.param)
    add_exits(model, EXITS[request.param], POOL, HIDDEN)
    settle_batchnorm(model)  # only the backbone's statistics: `forward` runs no head
    return model.eval()


def confidences(model, frame) -> dict[int, tuple[int, float]]:
    """(label, confidence) of every exit for `frame`, computed block by block."""
    x = normalise(frame)
    out = {}
    for block in range(model.num_blocks - 1):
        x = run_blocks(model, x, block, block + 1)
        if block in model.exit_points:
            out[block] = exit_check(model, x, block)
    return out


@pytest.mark.parametrize("size", [56, 28, 14, 7])
def test_exit_head_gives_logits_for_any_feature_map_size(size):
    head = exit_head(in_channels=24, pool=POOL, hidden=HIDDEN).eval()
    assert head(torch.rand(2, 24, size, size)).shape == (2, NUM_CLASSES)


def test_add_exits_builds_heads_for_the_channels_of_their_blocks(model):
    assert model.exit_points == EXITS[model.name]
    channels = block_channels(model)
    assert len(channels) == model.num_blocks - 1
    assert [channels[b] for b in EXITS[model.name]] == CHANNELS[model.name]
    for block, expected in zip(EXITS[model.name], CHANNELS[model.name], strict=True):
        assert model.exits[str(block)][1].in_channels == expected


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_add_exits_leaves_the_backbone_as_it_was(name, image):
    torch.manual_seed(SEED)
    model = build_model(name)  # in training mode, where a forward pass would move BatchNorm
    before = {key: value.clone() for key, value in model.state_dict().items()}
    add_exits(model, EXITS[name], POOL, HIDDEN)
    assert model.training
    after = model.state_dict()
    assert all(torch.equal(after[key], value) for key, value in before.items())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_add_exits_rejects_a_position_behind_the_last_feature_map(name):
    model = build_model(name)
    with pytest.raises(ValueError):
        add_exits(model, [model.num_blocks - 1], POOL, HIDDEN)


def test_checkpoint_with_exits_is_rebuilt_with_its_heads(model, frame, tmp_path):
    path = tmp_path / "exits.pt"
    torch.save(
        {
            "model": model.name,
            "state_dict": model.state_dict(),
            "exits": {"blocks": EXITS[model.name], "pool": POOL, "hidden": HIDDEN},
        },
        path,
    )
    torch.manual_seed(OTHER_SEED)  # so heads that kept their fresh weights would differ
    loaded = load_checkpoint(path)
    assert loaded.exit_points == EXITS[model.name]
    assert not loaded.training
    assert confidences(loaded, frame) == confidences(model, frame)
    assert torch.equal(loaded(normalise(frame)), model(normalise(frame)))


def test_with_tau_one_no_head_runs_and_the_result_is_that_of_run_head(model, frame):
    def fail(*_):
        raise AssertionError("an exit head ran although tau = 1.0")

    handles = [head.register_forward_hook(fail) for head in model.exits.values()]
    try:
        for k in range(model.num_blocks + 1):
            out, taken = run_head_exits(model, frame, k, TAU_NEVER)
            assert taken is None
            assert torch.equal(out, run_head(model, frame, k))
    finally:
        for handle in handles:
            handle.remove()


def test_only_exits_before_the_split_point_are_checked(model, frame):
    first = EXITS[model.name][0]
    expected = confidences(model, frame)
    for k in range(model.num_blocks + 1):
        out, taken = run_head_exits(model, frame, k, TAU_ALWAYS)
        if k <= first:  # the exit after block `first` needs that block, so k > first
            assert taken is None
            assert torch.equal(out, run_head(model, frame, k))
        else:
            assert taken == Exit(first, *expected[first])
            assert torch.equal(out, run_head(model, frame, first + 1))


def test_the_first_exit_that_is_confident_enough_answers(model, frame):
    expected = confidences(model, frame)
    n = model.num_blocks
    ordered = sorted(expected)
    for position, block in enumerate(ordered):
        earlier = [expected[b][1] for b in ordered[:position]]
        tau = expected[block][1]  # exactly its confidence: ">= tau" must take it
        if tau >= TAU_NEVER or any(c >= tau for c in earlier):
            continue
        _, taken = run_head_exits(model, frame, n, tau)
        assert taken == Exit(block, *expected[block])
    # Above every confidence nothing exits and the frame gets the final output.
    tau = max(c for _, c in expected.values()) + 1e-6
    assert tau < TAU_NEVER
    out, taken = run_head_exits(model, frame, n, tau)
    assert taken is None
    assert torch.equal(out, run_head(model, frame, n))


def test_forced_walk_takes_the_named_exit_or_none(model, frame):
    expected = confidences(model, frame)
    n = model.num_blocks
    for block in EXITS[model.name]:
        out, taken = run_head_forced(model, frame, n, block)
        assert taken == Exit(block, *expected[block])
        assert torch.equal(out, run_head(model, frame, block + 1))
    out, taken = run_head_forced(model, frame, n, None)
    assert taken is None
    assert torch.equal(out, run_head(model, frame, n))
    # An exit behind the split point is never reached.
    first = EXITS[model.name][0]
    assert run_head_forced(model, frame, first, first)[1] is None


def test_exit_walk_rejects_a_split_point_outside_the_model(model, frame):
    for k in (-1, model.num_blocks + 1):
        with pytest.raises(ValueError):
            run_head_exits(model, frame, k, TAU_ALWAYS)
