"""Fixtures shared by the tests: model inputs, models of ordinary scale, and a local edge."""

from collections.abc import Callable, Iterator

import pytest
import torch

from edgeinf.device.client import EdgeClient
from edgeinf.edge.server import InferenceServicer, start_server
from edgeinf.models.multi_exit import INPUT_SIZE, MODEL_NAMES, MultiExitModel, build_model
from edgeinf.profiler.layer_profile import raw_image

IMAGE_SEED = 0
BATCHNORM_SEED = 1
BATCHNORM_BATCH = 8  # images used to set the BatchNorm statistics
MODEL_SEED = 0
FRAME_SEED = 0
STOP_GRACE_S = 0


@pytest.fixture(scope="session")
def image() -> torch.Tensor:
    """One model input, shape (1, 3, 224, 224), values in [0, 1)."""
    generator = torch.Generator().manual_seed(IMAGE_SEED)
    return torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE, generator=generator)


@pytest.fixture(scope="session")
def frame() -> torch.Tensor:
    """One 8-bit camera frame, shape (1, 3, 224, 224), as a device holds it."""
    return raw_image(FRAME_SEED)


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


@pytest.fixture(scope="session")
def edge_models(settle_batchnorm) -> dict[str, MultiExitModel]:
    """Both models in eval mode, random weights, settled BatchNorm. Shared: do not change them."""
    torch.manual_seed(MODEL_SEED)
    models = {}
    for name in MODEL_NAMES:
        models[name] = build_model(name).eval()
        settle_batchnorm(models[name])
    return models


@pytest.fixture
def servicer(edge_models) -> Iterator[InferenceServicer]:
    """The edge's servicer with both models on the CPU."""
    servicer = InferenceServicer(edge_models.values(), device="cpu")
    yield servicer
    servicer.close()


@pytest.fixture
def edge_addr(servicer) -> Iterator[str]:
    """Address of an edge server on localhost that serves `servicer`."""
    server, port = start_server(port=0, host="localhost", servicer=servicer)
    yield f"localhost:{port}"
    server.stop(STOP_GRACE_S)


@pytest.fixture
def client(edge_addr) -> Iterator[EdgeClient]:
    with EdgeClient(edge_addr, device_id="device-1") as client:
        yield client
