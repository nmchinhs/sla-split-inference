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
    "edgeinf",
)


@pytest.mark.parametrize("name", REQUIRED_MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)
