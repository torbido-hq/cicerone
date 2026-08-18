"""Incremental event ingest (EventSource backends + micro-batch updater)."""

from __future__ import annotations

from cicerone.events.base import EventBackpressureError, EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.db import DbEventSource
from cicerone.events.redis_streams import RedisStreamsEventSource
from cicerone.events.registry import (
    build_event_source,
    register_event_source,
    registered_event_source_kinds,
)
from cicerone.events.s3 import S3EventSource
from cicerone.events.webhook import WebhookEventSource

__all__ = [
    "DbEventSource",
    "EventBackpressureError",
    "EventSource",
    "EventSourceHealth",
    "NormalizedEvent",
    "RedisStreamsEventSource",
    "S3EventSource",
    "WebhookEventSource",
    "build_event_source",
    "register_event_source",
    "registered_event_source_kinds",
]
