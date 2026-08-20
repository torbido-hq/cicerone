"""Checks the built wheel for console-script and dashboard static files."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from cicerone import __version__

DISTRIBUTION_NAME = "cicerone-recommender"
WHEEL_NAME = DISTRIBUTION_NAME.replace("-", "_")
CONSOLE_SCRIPT = "cicerone.cli:main"
REQUIRED_SUFFIXES = (
    "cicerone/static/tailwind.css",
    "cicerone/static/dashboard.js",
    "cicerone/templates/dashboard.html",
    "cicerone/templates/_status.html",
    "cicerone/templates/_recommendations.html",
    "cicerone/serve/python_detect.sh",
)


def parse_wheel_filename(filename: str) -> tuple[str, str, str | None]:
    """Return (name, version, build_tag) from a PEP 427 wheel filename."""
    if not filename.endswith(".whl"):
        raise ValueError(f"{filename}: not a wheel filename")
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        raise ValueError(f"{filename}: expected name-version[-build]-python-abi-platform.whl")
    name = parts[0]
    middle = parts[1:-3]
    if len(middle) == 1:
        return name, middle[0], None
    if len(middle) == 2 and middle[1][:1].isdigit():
        return name, middle[0], middle[1]
    raise ValueError(f"{filename}: cannot parse version from {filename[:-4]}")


def version_matches(wheel_version: str, requested: str) -> bool:
    return wheel_version == requested or wheel_version.startswith(f"{requested}+")


def select_wheel(
    dist_dir: str | Path = "dist",
    *,
    name: str = WHEEL_NAME,
    version: str = __version__,
) -> Path:
    directory = Path(dist_dir)
    wheels = sorted(directory.glob("*.whl"))
    found = [path.name for path in wheels]
    candidates: list[tuple[Path, str, str | None, tuple[str, str, str]]] = []
    for path in wheels:
        try:
            dist, ver, build = parse_wheel_filename(path.name)
        except ValueError:
            continue
        if dist != name or not version_matches(ver, version):
            continue
        raw_tags = path.name[:-4].split("-")[-3:]
        tags = (raw_tags[0], raw_tags[1], raw_tags[2])
        candidates.append((path, ver, build, tags))

    extra = f" (found: {', '.join(found)})" if found else ""
    if not candidates:
        raise ValueError(
            f"no {name} wheel matching version {version} "
            f"(PEP 440 local +… and numeric build tags ok) in {directory}{extra}"
        )

    tag_sets = {item[3] for item in candidates}
    if len(tag_sets) > 1:
        names = ", ".join(item[0].name for item in candidates)
        raise ValueError(f"multiple {name} {version} wheels with different tags in {directory}: {names}")

    candidates.sort(key=lambda item: (item[1] != version, item[2] is not None, item[0].name))
    return candidates[0][0]


def validate_wheel(wheel_path: str | Path) -> None:
    path = Path(wheel_path)
    name, version, _build = parse_wheel_filename(path.name)
    dist_info = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        entry_points = f"{dist_info}/entry_points.txt"
        if entry_points not in names:
            raise ValueError(f"{path}: missing {entry_points}")
        text = zf.read(entry_points).decode()
        if CONSOLE_SCRIPT not in text:
            raise ValueError(f"{path}: console script {CONSOLE_SCRIPT} not in entry_points.txt")
        for suffix in REQUIRED_SUFFIXES:
            if not any(member.endswith(suffix) for member in names):
                raise ValueError(f"{path}: missing {suffix}")


def validate_dist(
    dist_dir: str | Path = "dist",
    *,
    name: str = WHEEL_NAME,
    version: str = __version__,
) -> Path:
    wheel = select_wheel(dist_dir, name=name, version=version)
    validate_wheel(wheel)
    return wheel


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    dist_dir = args[0] if args else "dist"
    validate_dist(dist_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
