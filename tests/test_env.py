"""Sanity checks that the development environment is installed correctly."""

import importlib

import pytest

REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "grpc",
    "grpc_tools",
    "numpy",
    "pandas",
    "matplotlib",
    "yaml",
    "PIL",  # Pillow: comes with torchvision; encodes and decodes the JPEG of offload_jpeg
    "edgeinf",
)


@pytest.mark.parametrize("name", REQUIRED_MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)
