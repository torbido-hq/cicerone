"""kind → EventSource factory (internal; no external entry-points)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.events.base import EventSource
from cicerone.events.webhook import WebhookEventSource

_EventSourceFactory = Callable[[dict[str, Any]], EventSource]


def _db_source(options: dict[str, Any]) -> EventSource:
    from cicerone.events.db import DbEventSource

    return DbEventSource(options)


def _redis_source(options: dict[str, Any]) -> EventSource:
    from cicerone.events.redis_streams import RedisStreamsEventSource

    return RedisStreamsEventSource(options)


def _s3_source(options: dict[str, Any]) -> EventSource:
    from cicerone.events.s3 import S3EventSource

    return S3EventSource(options)


def _kafka_source(options: dict[str, Any]) -> EventSource:
    from cicerone.events.kafka import KafkaEventSource

    return KafkaEventSource(options)


def _rabbitmq_source(options: dict[str, Any]) -> EventSource:
    from cicerone.events.rabbitmq import RabbitMQEventSource

    return RabbitMQEventSource(options)


_EVENT_SOURCES: dict[str, _EventSourceFactory] = {
    "db": _db_source,
    "kafka": _kafka_source,
    "rabbitmq": _rabbitmq_source,
    "redis_streams": _redis_source,
    "s3": _s3_source,
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
