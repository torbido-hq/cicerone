"""kind → RecommendationPublisher factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.config.settings import PublishSettings
from cicerone.publish.base import RecommendationPublisher
from cicerone.publish.kafka import KafkaPublisher
from cicerone.publish.rabbitmq import RabbitMQPublisher

_PublisherFactory = Callable[[dict[str, Any]], RecommendationPublisher]

_PUBLISHERS: dict[str, _PublisherFactory] = {
    "kafka": KafkaPublisher,
    "rabbitmq": RabbitMQPublisher,
}


def build_publisher(publish: PublishSettings) -> RecommendationPublisher | None:
    if not publish.enabled:
        return None
    factory = _PUBLISHERS.get(publish.kind)
    if factory is None:
        raise ConfigError(f"publish.kind must be one of {sorted(_PUBLISHERS)}, got {publish.kind!r}")
    return factory(dict(publish.options))
