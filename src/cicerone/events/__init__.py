"""Incremental event ingest (EventSource backends + micro-batch updater)."""

from __future__ import annotations

from cicerone.events.base import EventBackpressureError, EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.registry import build_event_source, register_event_source

__all__ = [
    "EventBackpressureError",
    "EventSource",
    "EventSourceHealth",
    "NormalizedEvent",
    "build_event_source",
    "register_event_source",
]
