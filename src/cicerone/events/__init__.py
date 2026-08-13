"""Incremental event ingest (EventSource backends + micro-batch updater)."""

from __future__ import annotations

# Import backends so they register at package import time.
from cicerone.events import webhook as _webhook  # noqa: F401
from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.registry import build_event_source, register_event_source

__all__ = [
    "EventSource",
    "EventSourceHealth",
    "NormalizedEvent",
    "build_event_source",
    "register_event_source",
]
