"""Smoke test: the package and every subpackage import cleanly."""

import importlib

import pytest

SUBPACKAGES = ["data", "models", "training", "eval", "serving"]


def test_package_exposes_version() -> None:
    import antispoof

    assert isinstance(antispoof.__version__, str)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"antispoof.{name}")
