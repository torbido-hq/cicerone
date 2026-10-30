"""Incremental event ingest (EventSource backends + micro-batch updater)."""

from __future__ import annotations

from typing import Any

from cicerone.events.base import (
    EventBackpressureError,
    EventSource,
    EventSourceError,
    EventSourceHealth,
    NormalizedEvent,
)
from cicerone.lazy import lazy_attribute

__all__ = [
    "DbEventSource",
    "EventBackpressureError",
    "EventSource",
    "EventSourceError",
    "EventSourceHealth",
    "KafkaEventSource",
    "NormalizedEvent",
    "RabbitMQEventSource",
    "RedisStreamsEventSource",
    "S3EventSource",
    "WebhookEventSource",
    "build_event_source",
    "register_event_source",
    "registered_event_source_kinds",
]

_LAZY = {
    "DbEventSource": "cicerone.events.db",
    "KafkaEventSource": "cicerone.events.kafka",
    "RabbitMQEventSource": "cicerone.events.rabbitmq",
    "RedisStreamsEventSource": "cicerone.events.redis_streams",
    "S3EventSource": "cicerone.events.s3",
    "WebhookEventSource": "cicerone.events.webhook",
    "build_event_source": "cicerone.events.registry",
    "register_event_source": "cicerone.events.registry",
    "registered_event_source_kinds": "cicerone.events.registry",
}


def __getattr__(name: str) -> Any:
    return lazy_attribute(__name__, _LAZY, name)
