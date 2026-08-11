"""Serve mode package: read API over precomputed recommendations."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SERVE_API_DESCRIPTION",
    "SERVE_API_TITLE",
    "SERVE_API_VERSION",
    "_ItemsFilterCache",
    "_available_item_ids",
    "_configure_reader_item_filters",
    "_filter_recommendations",
    "_generated_at",
    "_start_refresh_loop",
    "create_app",
    "main",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from cicerone.serve import app

    return getattr(app, name)
