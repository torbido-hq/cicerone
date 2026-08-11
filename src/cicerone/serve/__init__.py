"""Serve mode package: read API over precomputed recommendations."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SERVE_API_DESCRIPTION",
    "SERVE_API_TITLE",
    "SERVE_API_VERSION",
    "create_app",
    "main",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from cicerone.serve import app

    return getattr(app, name)
