"""The dashboard's HTTP Basic Auth users file: a small TOML file mapping
username -> bcrypt password hash, managed via
`python -m cicerone.manage_dashboard_users` rather than hand-edited.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def load_users(path: str | Path) -> dict[str, str]:
    """Returns {username: bcrypt_hash}. A missing file returns {} rather
    than raising, since no users configured yet is a valid state.
    """
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
        "# Managed by `python -m cicerone.manage_dashboard_users` -- do not edit by hand.",
        "[users]",
    ]
    for username in sorted(users):
        lines.append(f'{username} = "{users[username]}"')
    file_path.write_text("\n".join(lines) + "\n")
