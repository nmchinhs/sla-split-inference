"""JPEG measurements for offload_jpeg (profiler/jpeg_profile.py), with stand-in models.

A stand-in model makes the right answer known: one that looks at brightness is not disturbed by
JPEG, and one that looks at fine detail is.
"""

import pytest
import torch

from edgeinf.models.multi_exit import INPUT_SIZE
from edgeinf.profiler.jpeg_profile import (
    choose_quality,
    predictions,
    quality_profile,
    time_device,
    time_edge,
)

RAW_IMAGE_BYTES = 150_550  # 3 x 224 x 224 bytes and a 22-byte header
DARK, BRIGHT = 30, 220  # grey levels on either side of the ImageNet mean
DETAIL = 6  # grey levels of the fine pattern: enough to see, little enough for JPEG to drop
LOW_QUALITY, HIGH_QUALITY = 10, 100
REQUESTS = 2
WARMUP = 1
BATCH = 4


class _Brightness(torch.nn.Module):
    """Class 1 if the normalised image is brighter than the average ImageNet image, else 0."""

    name = "brightness"
    num_blocks = 1

    @property
    def blocks(self) -> list:
        return [self.forward]  # the whole stand-in is one block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2, 3))
        return torch.stack([-mean, mean], dim=1)


class _Detail(_Brightness):
    """Class 1 if neighbouring pixels differ, that is if the image has fine detail, else 0."""

    name = "detail"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = (x[..., 1:] - x[..., :-1]).abs().mean(dim=(1, 2, 3)) - 0.02
        return torch.stack([-detail, detail], dim=1)


def _flat(level: int) -> torch.Tensor:
    return torch.full((3, INPUT_SIZE, INPUT_SIZE), level, dtype=torch.uint8)


def _striped(level: int) -> torch.Tensor:
    """A flat image with every other column slightly brighter."""
    image = _flat(level)
    image[..., ::2] += DETAIL
    return image


FRAMES = torch.stack([_flat(DARK), _flat(BRIGHT), _flat(BRIGHT), _flat(DARK)])
BRIGHTNESS_LABELS = [0, 1, 1, 0]
STRIPES = torch.stack([_striped(DARK), _flat(DARK), _striped(BRIGHT), _striped(DARK)])
DETAIL_LABELS = [1, 0, 1, 1]


@pytest.fixture(autouse=True)
def _inference_mode():
    with torch.inference_mode():
        yield


# ------------------------------------------------------------------------------ the rule


def test_the_lowest_quality_that_costs_less_than_half_a_point_is_chosen():
    assert choose_quality({50: 1.2, 75: 0.4, 90: 0.1, 95: 0.0}, max_drop_points=0.5) == 75
    assert choose_quality({95: 0.0, 90: 0.1, 75: 0.4, 50: 1.2}, max_drop_points=0.5) == 75


def test_exactly_half_a_point_is_not_less_than_half_a_point():
    assert choose_quality({50: 0.5, 75: 0.3}, max_drop_points=0.5) == 75


def test_a_quality_that_happens_to_gain_accuracy_qualifies():
    assert choose_quality({50: -0.2, 75: 0.0}, max_drop_points=0.5) == 50


def test_a_quality_is_judged_on_its_own_not_on_the_ones_above_it():
    # 75 costs too much here although 50 does not: the rule takes the lowest that qualifies.
    assert choose_quality({50: 0.4, 75: 0.6, 90: 0.1}, max_drop_points=0.5) == 50


def test_no_quality_good_enough_is_an_error_not_a_silent_choice():
    with pytest.raises(ValueError, match="0.5"):
        choose_quality({50: 2.0, 75: 0.9}, max_drop_points=0.5)


# ------------------------------------------------------------------------------ accuracy and size


def test_raw_frames_are_classified_after_normalisation():
    # Without normalisation both grey levels are positive and every frame would be class 1.
    labels, sizes = predictions(_Brightness(), FRAMES, None, "cpu", BATCH)
    assert labels == BRIGHTNESS_LABELS
    assert sizes == [RAW_IMAGE_BYTES] * len(FRAMES)


def test_jpeg_frames_are_classified_from_the_decoded_image():
    labels, sizes = predictions(_Brightness(), FRAMES, HIGH_QUALITY, "cpu", BATCH)
    assert labels == BRIGHTNESS_LABELS
    assert all(0 < size < RAW_IMAGE_BYTES for size in sizes)

    # Fine detail survives the highest quality and is gone at the lowest.
    assert predictions(_Detail(), STRIPES, None, "cpu", BATCH)[0] == DETAIL_LABELS
    assert predictions(_Detail(), STRIPES, HIGH_QUALITY, "cpu", BATCH)[0] == DETAIL_LABELS
    assert predictions(_Detail(), STRIPES, LOW_QUALITY, "cpu", BATCH)[0] == [0, 0, 0, 0]


def test_predictions_cover_every_frame_whatever_the_batch_size():
    one = predictions(_Detail(), STRIPES, HIGH_QUALITY, "cpu", 1)
    assert len(one[0]) == len(one[1]) == len(STRIPES)
    assert one == predictions(_Detail(), STRIPES, HIGH_QUALITY, "cpu", BATCH)
    # Four frames in batches of three: the last batch is not full and must not be dropped.
    assert one == predictions(_Detail(), STRIPES, HIGH_QUALITY, "cpu", len(STRIPES) - 1)


def test_profile_reports_accuracy_its_drop_and_the_bytes_per_quality():
    labels = [1, 0, 1, 0]  # the model is wrong on the last frame even with raw images
    profile = quality_profile(_Detail(), STRIPES, labels, (LOW_QUALITY, HIGH_QUALITY), "cpu", BATCH)

    assert profile["frames"] == len(STRIPES)
    assert profile["raw"] == {"accuracy": 0.75, "bytes": RAW_IMAGE_BYTES}
    high, low = profile["jpeg"][HIGH_QUALITY], profile["jpeg"][LOW_QUALITY]
    assert high["accuracy"] == 0.75
    assert high["accuracy_drop_points"] == 0.0
    assert high["labels_changed"] == 0
    # At the lowest quality every frame is class 0: frames 1 and 4 are then right.
    assert low["accuracy"] == 0.5
    assert low["accuracy_drop_points"] == 25.0
    assert low["labels_changed"] == 3
    for entry in (high, low):
        assert (
            entry["bytes_min"] <= entry["bytes_median"] <= entry["bytes_p95"] <= entry["bytes_max"]
        )
        assert entry["bytes_min"] <= entry["bytes_mean"] <= entry["bytes_max"]
    assert low["bytes_max"] < high["bytes_min"]


# ------------------------------------------------------------------------------ time


def test_device_times_cover_the_raw_frame_and_every_quality():
    times = time_device(FRAMES, (LOW_QUALITY, HIGH_QUALITY), REQUESTS, WARMUP, gap_ms=0.0)
    assert set(times) == {"raw", "jpeg"}
    assert set(times["jpeg"]) == {LOW_QUALITY, HIGH_QUALITY}
    for entry in (times["raw"], *times["jpeg"].values()):
        assert 0 < entry["device_ms"] <= entry["device_p95_ms"]


def test_edge_times_cover_the_raw_frame_and_every_quality():
    times = time_edge(_Brightness(), FRAMES, (LOW_QUALITY, HIGH_QUALITY), REQUESTS, WARMUP, "cpu")
    assert set(times) == {"raw", "jpeg"}
    assert set(times["jpeg"]) == {LOW_QUALITY, HIGH_QUALITY}
    for entry in (times["raw"], *times["jpeg"].values()):
        assert 0 < entry["edge_ms"] <= entry["edge_p95_ms"]
