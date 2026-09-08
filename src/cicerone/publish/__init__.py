"""Optional sidecar that publishes recommendations to Kafka or RabbitMQ."""

from __future__ import annotations

from cicerone.publish.base import RecommendationPublisher
from cicerone.publish.factory import build_publisher, registered_publish_kinds
from cicerone.publish.kafka import KafkaPublisher
from cicerone.publish.rabbitmq import RabbitMQPublisher

__all__ = [
    "KafkaPublisher",
    "RabbitMQPublisher",
    "RecommendationPublisher",
    "build_publisher",
    "registered_publish_kinds",
]
