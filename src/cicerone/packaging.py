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
)


def wheel_prefix(name: str = WHEEL_NAME, version: str = __version__) -> str:
    return f"{name}-{version}-"


def select_wheel(
    dist_dir: str | Path = "dist",
    *,
    name: str = WHEEL_NAME,
    version: str = __version__,
) -> Path:
    directory = Path(dist_dir)
    prefix = wheel_prefix(name, version)
    matches = sorted(path for path in directory.glob("*.whl") if path.name.startswith(prefix))
    if not matches:
        found = ", ".join(sorted(path.name for path in directory.glob("*.whl")))
        extra = f" (found: {found})" if found else ""
        raise ValueError(f"no wheel matching {prefix}*.whl in {directory}{extra}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"multiple wheels matching {prefix}*.whl in {directory}: {names}")
    return matches[0]


def validate_wheel(wheel_path: str | Path, *, name: str = WHEEL_NAME, version: str = __version__) -> None:
    path = Path(wheel_path)
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
    validate_wheel(wheel, name=name, version=version)
    return wheel


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    dist_dir = args[0] if args else "dist"
    validate_dist(dist_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
