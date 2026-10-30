"""Serve mode package: read API over precomputed recommendations."""

from __future__ import annotations

from typing import Any

from cicerone.lazy import lazy_attribute

__all__ = [
    "SERVE_API_DESCRIPTION",
    "SERVE_API_TITLE",
    "SERVE_API_VERSION",
    "create_app",
    "main",
]


_LAZY = dict.fromkeys(__all__, "cicerone.serve.app")


def __getattr__(name: str) -> Any:
    return lazy_attribute(__name__, _LAZY, name)
