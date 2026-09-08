"""kind → RecommendationPublisher factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.config.settings import PublishSettings, Settings
from cicerone.publish.base import RecommendationPublisher
from cicerone.publish.kafka import KafkaPublisher
from cicerone.publish.rabbitmq import RabbitMQPublisher

_PublisherFactory = Callable[[dict[str, Any]], RecommendationPublisher]

_PUBLISHERS: dict[str, _PublisherFactory] = {
    "kafka": KafkaPublisher,
    "rabbitmq": RabbitMQPublisher,
}


def build_publisher(settings: Settings | PublishSettings) -> RecommendationPublisher | None:
    publish = settings.publish if isinstance(settings, Settings) else settings
    if not publish.enabled:
        return None
    factory = _PUBLISHERS.get(publish.kind)
    if factory is None:
        raise ConfigError(f"publish.kind must be one of {sorted(_PUBLISHERS)}, got {publish.kind!r}")
    publisher = factory(dict(publish.options))
    publisher.connect()
    return publisher


def registered_publish_kinds() -> tuple[str, ...]:
    return tuple(sorted(_PUBLISHERS))


def build_publisher_from_kind(kind: str, options: dict[str, Any] | None = None) -> RecommendationPublisher:
    factory = _PUBLISHERS.get(kind.lower())
    if factory is None:
        raise ConfigError(f"publish.kind must be one of {sorted(_PUBLISHERS)}, got {kind!r}")
    publisher = factory(dict(options or {}))
    publisher.connect()
    return publisher
