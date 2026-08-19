"""Dashboard Basic Auth users file (username → bcrypt hash)."""

from __future__ import annotations

import tomllib
from pathlib import Path


def load_users(path: str | Path) -> dict[str, str]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("rb") as f:
        raw = tomllib.load(f)
    return dict(raw.get("users", {}))


def save_users(path: str | Path, users: dict[str, str]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Managed by `cicerone users` -- do not edit by hand.",
        "[users]",
    ]
    for username in sorted(users):
        lines.append(f'{username} = "{users[username]}"')
    file_path.write_text("\n".join(lines) + "\n")
