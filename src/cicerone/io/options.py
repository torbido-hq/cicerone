"""Small shared helper for validating backend "options" dicts (see
cicerone.config.IOSettings). Centralized here so every I/O backend reports
missing required options the same way.
"""

from __future__ import annotations

from typing import Any


def require_option(options: dict[str, Any], key: str, backend: str) -> Any:
    value = options.get(key)
    if value is None:
        raise RuntimeError(f"Missing required option '{key}' for backend {backend!r}")
    return value
