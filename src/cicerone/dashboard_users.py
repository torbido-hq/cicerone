"""The dashboard's HTTP Basic Auth users file: a small TOML file mapping
username -> bcrypt password hash, managed via
`python -m cicerone.manage_dashboard_users` (see that module for the
add/remove/list CLI) rather than hand-edited.

Deliberately separate from cicerone.config/cicerone.toml: unlike every
other secret in this repo (a single shared token per HTTP surface, resolved
from "${ENV_VAR}"), the dashboard is meant for a handful of named people to
log in with their own username/password via a browser -- a bcrypt hash
isn't a secret that needs an env var, and a small standalone file is what
the management CLI reads and writes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def load_users(path: str | Path) -> dict[str, str]:
    """Returns {username: bcrypt_hash}. A missing file means no users are
    configured yet -- returns {} rather than raising, so cicerone.dashboard
    can give a clear "no users configured" error instead of a confusing
    file-not-found one."""
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
