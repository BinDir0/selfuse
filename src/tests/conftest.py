"""Shared pytest configuration for EgoVLA tests."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: requires CUDA GPU")
    config.addinivalue_line("markers", "real_model: requires real model weights on disk")
