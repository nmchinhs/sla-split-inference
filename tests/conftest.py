"""Fixtures shared by the model tests."""

from collections.abc import Callable

import pytest
import torch

from edgeinf.models.multi_exit import INPUT_SIZE

IMAGE_SEED = 0
BATCHNORM_SEED = 1
BATCHNORM_BATCH = 8  # images used to set the BatchNorm statistics


@pytest.fixture(scope="session")
def image() -> torch.Tensor:
    """One model input, shape (1, 3, 224, 224), values in [0, 1)."""
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    return torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE, generator=generator)


@pytest.fixture(scope="session")
def settle_batchnorm() -> Callable[[torch.nn.Module], None]:
    """A function that sets a model's BatchNorm statistics from one batch, as training would.

    A freshly built model in eval mode normalises nothing (running mean 0, variance 1), and
    MobileNetV2's activations then shrink from about 1 after the first block to about 1e-10 at
    the logits. At that scale a comparison with an absolute tolerance passes for any values, and
    16-bit floats underflow. With statistics from a batch, activations stay near 1 as they do in
    a trained model.
    """
    generator = torch.Generator().manual_seed(BATCHNORM_SEED)
    batch = torch.rand(BATCHNORM_BATCH, 3, INPUT_SIZE, INPUT_SIZE, generator=generator)

    def settle(model: torch.nn.Module) -> None:
        # Resets the running statistics, averages them over the batch, restores eval mode.
        torch.optim.swa_utils.update_bn([batch], model)

    return settle
