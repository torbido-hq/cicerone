"""Packaging metadata for the PyPI distribution."""

from __future__ import annotations

import re
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
    assert extras["kafka"]["file"] == ["requirements-kafka.txt"]
    assert extras["rabbitmq"]["file"] == ["requirements-rabbitmq.txt"]
    root = _PYPROJECT.parent
    assert (root / "requirements-redis.txt").is_file()
    assert (root / "requirements-sequential.txt").is_file()
    assert (root / "requirements-kafka.txt").is_file()
    assert (root / "requirements-rabbitmq.txt").is_file()


def test_console_script_entry_point():
    scripts = _project()["project"]["scripts"]
    assert scripts["cicerone"] == "cicerone.cli:main"


def test_validate_wheel_requires_entry_point_and_static_files(tmp_path):
    import zipfile

    from cicerone.packaging import (
        REQUIRED_SUFFIXES,
        WHEEL_NAME,
        main,
        parse_wheel_filename,
        select_wheel,
        validate_dist,
        validate_wheel,
    )

    version = __version__
    dist_info = f"{WHEEL_NAME}-{version}.dist-info"

    def _write_wheel(path, *, files: dict[str, str]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

    def _valid_files(*, ver: str = version) -> dict[str, str]:
        ep = f"{WHEEL_NAME}-{ver}.dist-info/entry_points.txt"
        files = {ep: "[console_scripts]\ncicerone = cicerone.cli:main\n"}
        for suffix in REQUIRED_SUFFIXES:
            files[suffix] = "ok"
        return files

    wheel = tmp_path / f"{WHEEL_NAME}-{version}-py3-none-any.whl"
    decoy = tmp_path / "other_pkg-1.0.0-py3-none-any.whl"
    _write_wheel(wheel, files=_valid_files())
    _write_wheel(decoy, files={"other_pkg-1.0.0.dist-info/METADATA": "Name: other"})

    validate_wheel(wheel)
    assert select_wheel(tmp_path) == wheel
    assert validate_dist(tmp_path) == wheel
    assert main([str(tmp_path)]) == 0
    assert parse_wheel_filename(wheel.name) == (WHEEL_NAME, version, None)
    with pytest.raises(ValueError, match="not a wheel filename"):
        parse_wheel_filename("not-a-wheel")
    with pytest.raises(ValueError, match="cannot parse version"):
        parse_wheel_filename(f"{WHEEL_NAME}-{version}-py3-none-any-extra.whl")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no cicerone_recommender wheel matching version"):
        validate_dist(empty)

    wrong_version = tmp_path / "wrong"
    wrong_version.mkdir()
    _write_wheel(
        wrong_version / f"{WHEEL_NAME}-0.0.0-py3-none-any.whl",
        files=_valid_files(ver="0.0.0"),
    )
    with pytest.raises(ValueError, match="no cicerone_recommender wheel matching version"):
        select_wheel(wrong_version)

    patch_neighbor = tmp_path / "neighbor"
    patch_neighbor.mkdir()
    _write_wheel(
        patch_neighbor / f"{WHEEL_NAME}-{version}0-py3-none-any.whl",
        files=_valid_files(ver=f"{version}0"),
    )
    with pytest.raises(ValueError, match="no cicerone_recommender wheel matching version"):
        select_wheel(patch_neighbor)

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    local_ver = f"{version}+ci.1"
    local_wheel = local_dir / f"{WHEEL_NAME}-{local_ver}-py3-none-any.whl"
    _write_wheel(local_wheel, files=_valid_files(ver=local_ver))
    assert select_wheel(local_dir) == local_wheel
    validate_wheel(local_wheel)

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    build_wheel = build_dir / f"{WHEEL_NAME}-{version}-1-py3-none-any.whl"
    _write_wheel(build_wheel, files=_valid_files())
    assert select_wheel(build_dir) == build_wheel
    assert parse_wheel_filename(build_wheel.name) == (WHEEL_NAME, version, "1")

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    exact = mixed / f"{WHEEL_NAME}-{version}-py3-none-any.whl"
    local_also = mixed / f"{WHEEL_NAME}-{version}+dev-py3-none-any.whl"
    _write_wheel(exact, files=_valid_files())
    _write_wheel(local_also, files=_valid_files(ver=f"{version}+dev"))
    _write_wheel(mixed / "not-a-real.whl", files={"x": "y"})
    assert select_wheel(mixed) == exact

    no_ep = tmp_path / "no_ep"
    no_ep.mkdir()
    no_ep_wheel = no_ep / f"{WHEEL_NAME}-{version}-py3-none-any.whl"
    _write_wheel(no_ep_wheel, files={k: v for k, v in _valid_files().items() if "entry_points" not in k})
    with pytest.raises(ValueError, match="entry_points.txt"):
        validate_wheel(no_ep_wheel)

    dupes = tmp_path / "dupes"
    dupes.mkdir()
    _write_wheel(dupes / f"{WHEEL_NAME}-{version}-py3-none-any.whl", files=_valid_files())
    _write_wheel(dupes / f"{WHEEL_NAME}-{version}-cp311-none-any.whl", files=_valid_files())
    with pytest.raises(ValueError, match="different tags"):
        select_wheel(dupes)

    nameless = tmp_path / "bad.whl"
    _write_wheel(nameless, files={"cicerone/static/tailwind.css": "x"})
    with pytest.raises(ValueError, match="expected name-version"):
        validate_wheel(nameless)

    missing_script = tmp_path / "missing_script"
    missing_script.mkdir()
    missing_path = missing_script / f"{WHEEL_NAME}-{version}-py3-none-any.whl"
    files = _valid_files()
    files[f"{dist_info}/entry_points.txt"] = "[console_scripts]\n"
    _write_wheel(missing_path, files=files)
    with pytest.raises(ValueError, match="console script"):
        validate_wheel(missing_path)

    for suffix in REQUIRED_SUFFIXES:
        missing_dir = tmp_path / f"missing-{suffix.replace('/', '_')}"
        missing_dir.mkdir()
        missing = missing_dir / f"{WHEEL_NAME}-{version}-py3-none-any.whl"
        files = {k: v for k, v in _valid_files().items() if k != suffix}
        _write_wheel(missing, files=files)
        with pytest.raises(ValueError, match=re.escape(suffix)):
            validate_wheel(missing)


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
        "requirements-kafka.txt",
        "requirements-rabbitmq.txt",
        "LICENSE",
        "python_detect.sh",
    ):
        assert name in text
