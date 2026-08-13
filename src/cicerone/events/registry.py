"""kind → EventSource factory registry (internal; no external entry-points)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.events.base import EventSource

_EventSourceFactory = Callable[[dict[str, Any]], EventSource]

_REGISTRY: dict[str, _EventSourceFactory] = {}


def register_event_source(kind: str, factory: _EventSourceFactory) -> None:
    key = kind.lower()
    if key in _REGISTRY:
        raise ValueError(f"Event source kind already registered: {key!r}")
    _REGISTRY[key] = factory


def registered_event_source_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_event_source(kind: str, options: dict[str, Any] | None = None) -> EventSource:
    key = kind.lower()
    try:
        factory = _REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(registered_event_source_kinds()) or "(none)"
        raise ValueError(f"Unknown events kind: {kind!r}; registered: {known}") from exc
    return factory(dict(options or {}))
