"""Incremental event ingest (EventSource backends + micro-batch updater)."""

from __future__ import annotations

import importlib
from typing import Any

from cicerone.events.base import EventBackpressureError, EventSource, EventSourceHealth, NormalizedEvent

__all__ = [
    "DbEventSource",
    "EventBackpressureError",
    "EventSource",
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

_LAZY: dict[str, tuple[str, str]] = {
    "DbEventSource": ("cicerone.events.db", "DbEventSource"),
    "KafkaEventSource": ("cicerone.events.kafka", "KafkaEventSource"),
    "RabbitMQEventSource": ("cicerone.events.rabbitmq", "RabbitMQEventSource"),
    "RedisStreamsEventSource": ("cicerone.events.redis_streams", "RedisStreamsEventSource"),
    "S3EventSource": ("cicerone.events.s3", "S3EventSource"),
    "WebhookEventSource": ("cicerone.events.webhook", "WebhookEventSource"),
    "build_event_source": ("cicerone.events.registry", "build_event_source"),
    "register_event_source": ("cicerone.events.registry", "register_event_source"),
    "registered_event_source_kinds": ("cicerone.events.registry", "registered_event_source_kinds"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value
