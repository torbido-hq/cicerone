"""Publish per-user recommendation JSON to a Kafka topic."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from cicerone.config.constants import ConfigError
from cicerone.kafka_options import kafka_client_config, kafka_timeout_seconds, require_nonempty_str
from cicerone.publish.base import PublishError
from cicerone.publish.payload import user_recommendation_messages

logger = logging.getLogger(__name__)

_PREFIX = "publish.options"


def validate_kafka_publish_options(options: dict[str, Any]) -> None:
    kafka_client_config(options, prefix=_PREFIX)
    require_nonempty_str(options, "topic", prefix=_PREFIX)


def _missing_extra() -> ConfigError:
    return ConfigError(
        'publish.kind = "kafka" requires the confluent-kafka package; '
        "install with: pip install 'cicerone-recommender[kafka]'"
    )


class KafkaPublisher:
    def __init__(self, options: dict[str, Any]):
        validate_kafka_publish_options(options)
        self._conf = kafka_client_config(options, prefix=_PREFIX)
        self._timeout_seconds = kafka_timeout_seconds(options, prefix=_PREFIX)
        self._topic = require_nonempty_str(options, "topic", prefix=_PREFIX)
        self._producer: Any | None = None

    def connect(self) -> None:
        if self._producer is not None:
            return
        try:
            from confluent_kafka import Producer
        except ImportError as exc:
            raise _missing_extra() from exc
        producer = None
        try:
            producer = Producer(self._conf)
            producer.list_topics(timeout=self._timeout_seconds)
        except Exception as exc:
            if producer is not None:
                try:
                    producer.flush(self._timeout_seconds)
                except Exception:
                    logger.exception("Kafka publisher flush after connect failure")
            raise ConfigError(f"publish.options.bootstrap_servers is unreachable: {exc}") from exc
        self._producer = producer

    def publish(self, df: pd.DataFrame) -> None:
        producer = self._require()
        errors: list[str] = []

        def on_delivery(err: object, _msg: object) -> None:
            if err is not None:
                errors.append(str(err))

        messages = user_recommendation_messages(df)
        try:
            for user_id, body, _message_id in messages:
                producer.produce(
                    self._topic,
                    value=body,
                    key=user_id.encode("utf-8"),
                    on_delivery=on_delivery,
                )
            remaining = producer.flush(self._timeout_seconds)
        except Exception as exc:
            raise PublishError(f"Kafka publish failed: {exc}") from exc
        if remaining:
            raise PublishError(f"Kafka publish timed out with {remaining} message(s) in queue")
        if errors:
            raise PublishError(f"Kafka publish delivery failed: {errors[0]}")

    def close(self) -> None:
        producer = self._producer
        self._producer = None
        if producer is None:
            return
        try:
            producer.flush(self._timeout_seconds)
        except Exception as exc:
            raise PublishError(f"Kafka publisher flush on close failed: {exc}") from exc

    def _require(self) -> Any:
        if self._producer is None:
            raise PublishError("KafkaPublisher is not connected")
        return self._producer
