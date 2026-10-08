"""Layer profile (profiler/layer_profile.py): structure and sizes, with very few timed runs."""

import pytest
import torch

from edgeinf.common.serialize import serialize
from edgeinf.data.imagenette import MEAN, STD, normalise
from edgeinf.models.multi_exit import INPUT_SIZE, MODEL_NAMES, build_model
from edgeinf.models.split import run_blocks
from edgeinf.profiler.layer_profile import (
    device_step,
    edge_step,
    profile_device,
    profile_edge,
    raw_image,
)

SEED = 0
REQUESTS = 2
WARMUP = 1
GAP_MS = 0.0
RAW_IMAGE_BYTES = 150_550  # 3 x 224 x 224 bytes and a 22-byte header
CONFIDENCE_TOLERANCE = 0.01  # relative; 16-bit transfer moves the output a little


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request):
    torch.manual_seed(SEED)
    return build_model(request.param).eval()


def test_normalise_matches_the_dataset_transform():
    image = raw_image(SEED)
    assert image.dtype == torch.uint8 and image.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    expected = (image.float() / 255 - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(
        1, 3, 1, 1
    )
    assert torch.allclose(normalise(image), expected, atol=1e-6)
    assert normalise(image).dtype == torch.float32


def test_every_split_point_gives_the_answer_of_the_whole_model(model, settle_batchnorm):
    settle_batchnorm(model)  # activations of ordinary size, as in a trained model
    frame = raw_image(SEED)
    n = model.num_blocks
    with torch.inference_mode():
        confidence, label = model(normalise(frame)).softmax(1).max(1)
        assert device_step(model, frame, n) == int(label)
        for k in range(n):
            got_label, got_confidence = edge_step(model, device_step(model, frame, k), k, "cpu")
            assert got_label == int(label), f"{model.name}: k={k}"
            assert got_confidence == pytest.approx(float(confidence), rel=CONFIDENCE_TOLERANCE)


def test_device_profile_has_a_time_for_every_split_point_and_the_real_sizes(model):
    n = model.num_blocks
    profile = profile_device(model, REQUESTS, WARMUP, GAP_MS, SEED)
    assert len(profile["device_ms"]) == len(profile["device_p95_ms"]) == n + 1
    assert all(ms > 0 for ms in profile["device_ms"])
    assert all(
        p95 >= ms for ms, p95 in zip(profile["device_ms"], profile["device_p95_ms"], strict=True)
    )

    # bytes[k] is what the device sends at split point k: the raw image at 0, nothing at n.
    assert len(profile["bytes"]) == n + 1
    assert profile["bytes"][0] == RAW_IMAGE_BYTES
    assert profile["bytes"][n] == 0
    with torch.inference_mode():
        x = normalise(raw_image(SEED))
        for k in range(1, n):
            assert profile["bytes"][k] == len(serialize(run_blocks(model, x, 0, k), fp16=True))


def test_edge_profile_has_a_time_for_every_split_point_it_can_receive(model):
    profile = profile_edge(model, REQUESTS, WARMUP, "cpu", SEED)
    assert len(profile["edge_ms"]) == len(profile["edge_p95_ms"]) == model.num_blocks
    assert all(ms > 0 for ms in profile["edge_ms"])


def test_running_the_whole_model_costs_more_than_its_first_block(model):
    profile = profile_device(model, REQUESTS, WARMUP, GAP_MS, SEED)
    assert profile["device_ms"][model.num_blocks] > profile["device_ms"][1]
