"""Packaging metadata for the PyPI distribution."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

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


def test_console_script_entry_point():
    scripts = _project()["project"]["scripts"]
    assert scripts["cicerone"] == "cicerone.cli:main"


def test_validate_wheel_requires_entry_point_and_static_files(tmp_path):
    import zipfile

    from cicerone.packaging import main, validate_dist, validate_wheel

    wheel = tmp_path / "cicerone_recommender-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(
            "cicerone_recommender-0.0.0.dist-info/entry_points.txt",
            "[console_scripts]\ncicerone = cicerone.cli:main\n",
        )
        zf.writestr("cicerone/static/tailwind.css", "/* css */")
        zf.writestr("cicerone/templates/dashboard.html", "<html></html>")

    validate_wheel(wheel)
    assert validate_dist(tmp_path) == wheel
    assert main([str(tmp_path)]) == 0

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no wheel"):
        validate_dist(empty)

    bad = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("cicerone/static/tailwind.css", "x")
    with pytest.raises(ValueError, match="entry_points.txt"):
        validate_wheel(bad)

    missing_script = tmp_path / "noscript.whl"
    with zipfile.ZipFile(missing_script, "w") as zf:
        zf.writestr("cicerone_recommender-0.0.0.dist-info/entry_points.txt", "[console_scripts]\n")
        zf.writestr("cicerone/static/tailwind.css", "x")
        zf.writestr("cicerone/templates/dashboard.html", "x")
    with pytest.raises(ValueError, match="console script"):
        validate_wheel(missing_script)

    missing_css = tmp_path / "nocss.whl"
    with zipfile.ZipFile(missing_css, "w") as zf:
        zf.writestr(
            "cicerone_recommender-0.0.0.dist-info/entry_points.txt",
            "cicerone.cli:main",
        )
        zf.writestr("cicerone/templates/dashboard.html", "x")
    with pytest.raises(ValueError, match="tailwind.css"):
        validate_wheel(missing_css)

    missing_html = tmp_path / "nohtml.whl"
    with zipfile.ZipFile(missing_html, "w") as zf:
        zf.writestr(
            "cicerone_recommender-0.0.0.dist-info/entry_points.txt",
            "cicerone.cli:main",
        )
        zf.writestr("cicerone/static/tailwind.css", "x")
    with pytest.raises(ValueError, match="dashboard.html"):
        validate_wheel(missing_html)


def test_packaging_main_defaults_to_dist(monkeypatch, tmp_path):
    from cicerone import packaging

    called: list[str] = []
    monkeypatch.setattr(packaging, "validate_dist", lambda dist_dir: called.append(dist_dir))
    monkeypatch.setattr(packaging.sys, "argv", ["cicerone.packaging"])
    assert packaging.main() == 0
    assert called == ["dist"]
    assert packaging.main([str(tmp_path)]) == 0
    assert called == ["dist", str(tmp_path)]


def test_sdist_includes_requirement_pins():
    text = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for name in (
        "requirements.txt",
        "requirements-redis.txt",
        "requirements-sequential.txt",
        "LICENSE",
    ):
        assert name in text
