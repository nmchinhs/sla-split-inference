"""Block structure of the backbones (models/multi_exit.py), and the hook for exit heads."""

import pytest
import torch
import torchvision

from edgeinf.models.multi_exit import MODEL_NAMES, NUM_CLASSES, build_model, load_checkpoint

SEED = 0
OTHER_SEED = 1
# torch.allclose accepts differences up to 1e-8 whatever the scale, so the reference logits must
# be far larger than that for the comparison to mean anything.
MIN_LOGIT = 1e-2

# (channels, height) of the output of every block but the last, at 224x224; width equals height.
# The last block ends in the classifier and returns NUM_CLASSES logits.
FEATURE_SHAPES = {
    "mobilenetv2": [
        (32, 112),
        (16, 112),
        (24, 56),
        (24, 56),
        (32, 28),
        (32, 28),
        (32, 28),
        (64, 14),
        (64, 14),
        (64, 14),
        (64, 14),
        (96, 14),
        (96, 14),
        (96, 14),
        (160, 7),
        (160, 7),
        (160, 7),
        (320, 7),
        (1280, 7),
    ],
    "resnet18": [
        (64, 56),
        (64, 56),
        (64, 56),
        (128, 28),
        (128, 28),
        (256, 14),
        (256, 14),
        (512, 7),
        (512, 7),
    ],
}
TORCHVISION_BUILDERS = {
    "mobilenetv2": torchvision.models.mobilenet_v2,
    "resnet18": torchvision.models.resnet18,
}
EXIT_AFTER = 3  # an early block that exists in both models
EXIT_HEAD_IN_FEATURES = 4  # the dummy head is registered but never run


@pytest.fixture(autouse=True)
def _inference_mode():
    with torch.inference_mode():
        yield


def _dummy_head() -> torch.nn.Module:
    return torch.nn.Linear(EXIT_HEAD_IN_FEATURES, NUM_CLASSES)


def test_model_names_match_the_project_models():
    assert set(MODEL_NAMES) == set(FEATURE_SHAPES)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="alexnet"):
        build_model("alexnet")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_blocks_are_ordered_and_have_the_expected_output_shapes(name, image):
    model = build_model(name).eval()
    assert model.name == name
    assert model.num_blocks == len(FEATURE_SHAPES[name]) + 1

    x = image
    for block, (channels, size) in zip(model.blocks[:-1], FEATURE_SHAPES[name], strict=True):
        x = block(x)
        assert x.shape == (1, channels, size, size)
    assert model.blocks[-1](x).shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_blocks_compute_the_torchvision_architecture(name, image, settle_batchnorm):
    torch.manual_seed(SEED)
    reference = TORCHVISION_BUILDERS[name](weights=None, num_classes=NUM_CLASSES).eval()
    torch.manual_seed(SEED)
    model = build_model(name).eval()

    # Same seed and same constructor, so the weights are identical; only the grouping differs.
    for ours, theirs in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(ours, theirs)
    settle_batchnorm(reference)
    settle_batchnorm(model)
    expected = reference(image)
    assert expected.abs().max() > MIN_LOGIT
    assert torch.allclose(model(image), expected)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_weights_are_random_and_fixed_by_the_seed(name):
    def weights(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.cat([p.flatten() for p in build_model(name).parameters()])

    assert torch.equal(weights(SEED), weights(SEED))
    assert not torch.equal(weights(SEED), weights(OTHER_SEED))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_new_model_has_no_exits(name):
    model = build_model(name)
    assert model.exit_points == []
    assert all(key.startswith("blocks.") for key in model.state_dict())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_add_exit_registers_a_head_after_a_block(name, image):
    torch.manual_seed(SEED)
    model = build_model(name).eval()
    before = model(image)

    model.add_exit(EXIT_AFTER + 1, _dummy_head())
    model.add_exit(EXIT_AFTER, _dummy_head())

    assert model.exit_points == [EXIT_AFTER, EXIT_AFTER + 1]
    # The final output never goes through an exit head.
    assert torch.equal(model(image), before)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_backbone_checkpoint_loads_after_exits_are_added(name):
    checkpoint = build_model(name).state_dict()
    model = build_model(name)
    model.add_exit(EXIT_AFTER, _dummy_head())

    result = model.load_state_dict(checkpoint, strict=False)

    assert result.unexpected_keys == []
    assert result.missing_keys
    assert all(key.startswith(f"exits.{EXIT_AFTER}.") for key in result.missing_keys)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_add_exit_rejects_invalid_positions(name):
    model = build_model(name)
    last = model.num_blocks - 1
    # The last block already returns the final output, so no exit can follow it.
    for after_block in (-1, last, last + 1):
        with pytest.raises(ValueError):
            model.add_exit(after_block, _dummy_head())

    model.add_exit(EXIT_AFTER, _dummy_head())
    with pytest.raises(ValueError):
        model.add_exit(EXIT_AFTER, _dummy_head())


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_load_checkpoint_rebuilds_the_trained_model_ready_for_inference(
    name, image, settle_batchnorm, tmp_path
):
    torch.manual_seed(SEED)
    trained = build_model(name).eval()
    settle_batchnorm(trained)  # BatchNorm statistics are part of what a checkpoint must restore
    expected = trained(image)
    assert expected.abs().max() > MIN_LOGIT
    # The file layout that scripts/train_backbone.py writes.
    path = tmp_path / "trained.pt"
    torch.save({"model": name, "state_dict": trained.state_dict(), "epochs": 1}, path)

    torch.manual_seed(OTHER_SEED)  # so a model that kept its fresh weights would differ
    model = load_checkpoint(path)

    assert model.name == name
    assert not model.training
    assert torch.equal(model(image), expected)


def test_load_checkpoint_refuses_a_file_that_is_not_one_of_ours(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save(build_model("resnet18").state_dict(), path)
    with pytest.raises(ValueError, match="train_backbone"):
        load_checkpoint(path)
