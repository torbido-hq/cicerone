"""Packaging metadata for the PyPI distribution."""

from __future__ import annotations

import tomllib
from pathlib import Path

from cicerone import __version__

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"


def _project() -> dict:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_distribution_name_is_not_taken_cicerone():
    project = _project()["project"]
    assert project["name"] == "cicerone-recommender"
    assert project["requires-python"] == ">=3.11,<3.12"
    assert "version" in project["dynamic"]
    assert "dependencies" in project["dynamic"]
    assert "optional-dependencies" in project["dynamic"]


def test_version_is_read_from_package_attr():
    dynamic = _project()["tool"]["setuptools"]["dynamic"]
    assert dynamic["version"]["attr"] == "cicerone.__version__"
    assert __version__
    assert __version__.count(".") >= 2


def test_optional_extras_match_requirements_files():
    extras = _project()["tool"]["setuptools"]["dynamic"]["optional-dependencies"]
    assert extras["redis"]["file"] == ["requirements-redis.txt"]
    assert extras["sequential"]["file"] == ["requirements-sequential.txt"]
    root = _PYPROJECT.parent
    assert (root / "requirements-redis.txt").is_file()
    assert (root / "requirements-sequential.txt").is_file()


def test_sdist_includes_requirement_pins():
    text = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for name in (
        "requirements.txt",
        "requirements-redis.txt",
        "requirements-sequential.txt",
        "LICENSE",
    ):
        assert name in text
