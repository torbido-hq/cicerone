"""JSON reads for event source position files."""

from __future__ import annotations

import json
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
