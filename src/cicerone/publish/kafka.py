"""Publish per-user recommendation JSON to a Kafka topic."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from cicerone.config.constants import ConfigError
from cicerone.kafka_options import kafka_client_config, require_nonempty_str
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
        self._topic = require_nonempty_str(options, "topic", prefix=_PREFIX)
        self._producer: Any | None = None

    def connect(self) -> None:
        try:
            from confluent_kafka import Producer
        except ImportError as exc:
            raise _missing_extra() from exc
        producer = Producer(self._conf)
        try:
            producer.list_topics(timeout=10)
        except Exception as exc:
            try:
                producer.flush(1)
            except Exception:
                logger.exception("Kafka publisher flush after connect failure")
            raise ConfigError(f"publish.options.bootstrap_servers is unreachable: {exc}") from exc
        self._producer = producer

    def publish(self, df: pd.DataFrame) -> None:
        producer = self._require()
        for user_id, body in user_recommendation_messages(df):
            producer.produce(self._topic, value=body, key=user_id.encode("utf-8"))
        remaining = producer.flush(10)
        if remaining:
            raise RuntimeError(f"Kafka publish timed out with {remaining} message(s) in queue")

    def close(self) -> None:
        producer = self._producer
        self._producer = None
        if producer is None:
            return
        try:
            producer.flush(10)
        except Exception:
            logger.exception("Kafka publisher flush on close failed")

    def _require(self) -> Any:
        if self._producer is None:
            raise RuntimeError("KafkaPublisher is not connected")
        return self._producer
