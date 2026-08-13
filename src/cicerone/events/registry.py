"""kind → EventSource factory (internal; no external entry-points)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.events.base import EventSource
from cicerone.events.webhook import WebhookEventSource

_EventSourceFactory = Callable[[dict[str, Any]], EventSource]

_EVENT_SOURCES: dict[str, _EventSourceFactory] = {
    "webhook": WebhookEventSource,
}


def register_event_source(kind: str, factory: _EventSourceFactory) -> None:
    """Register a built-in backend (tests / future kinds). Rejects duplicates."""
    key = kind.lower()
    if key in _EVENT_SOURCES:
        raise ValueError(f"Event source kind already registered: {key!r}")
    _EVENT_SOURCES[key] = factory


def registered_event_source_kinds() -> tuple[str, ...]:
    return tuple(sorted(_EVENT_SOURCES))


def build_event_source(kind: str, options: dict[str, Any] | None = None) -> EventSource:
    key = kind.lower()
    try:
        factory = _EVENT_SOURCES[key]
    except KeyError as exc:
        known = ", ".join(registered_event_source_kinds()) or "(none)"
        raise ValueError(f"Unknown events kind: {kind!r}; registered: {known}") from exc
    return factory(dict(options or {}))
