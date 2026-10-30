"""JSON reads and writes for event source position files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_json_object_file(path: Path) -> dict[str, Any] | None:
    """The JSON object in ``path``, or None when it does not exist; ValueError when it is not an object."""
    # Read directly: is_file() reports an inaccessible file as absent and would restart from a stale position.
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} root must be an object")
    return raw


def write_json_object_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    # Sync the directory after the rename so the rename itself survives a crash.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
