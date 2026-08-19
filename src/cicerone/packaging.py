"""Checks the built wheel for console-script and dashboard static files."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

CONSOLE_SCRIPT = "cicerone.cli:main"
STATIC_CSS = "cicerone/static/tailwind.css"
DASHBOARD_TEMPLATE = "cicerone/templates/dashboard.html"


def validate_wheel(wheel_path: str | Path) -> None:
    path = Path(wheel_path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        entry_points = [name for name in names if name.endswith("entry_points.txt")]
        if not entry_points:
            raise ValueError(f"{path}: missing entry_points.txt")
        text = zf.read(entry_points[0]).decode()
        if CONSOLE_SCRIPT not in text:
            raise ValueError(f"{path}: console script {CONSOLE_SCRIPT} not in entry_points.txt")
        for suffix in (STATIC_CSS, DASHBOARD_TEMPLATE):
            if not any(name.endswith(suffix) for name in names):
                raise ValueError(f"{path}: missing {suffix}")


def validate_dist(dist_dir: str | Path = "dist") -> Path:
    wheels = sorted(Path(dist_dir).glob("*.whl"))
    if not wheels:
        raise ValueError(f"no wheel in {dist_dir}")
    wheel = wheels[0]
    validate_wheel(wheel)
    return wheel


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    dist_dir = args[0] if args else "dist"
    validate_dist(dist_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
