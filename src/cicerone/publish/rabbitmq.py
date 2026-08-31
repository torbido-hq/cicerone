"""Publish per-user recommendation JSON to a RabbitMQ queue or exchange."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from cicerone.amqp_options import optional_exchange, optional_routing_key, require_amqp_url, require_queue
from cicerone.config.constants import ConfigError
from cicerone.publish.payload import user_recommendation_messages

logger = logging.getLogger(__name__)

_PREFIX = "publish.options"


def validate_rabbitmq_publish_options(options: dict[str, Any]) -> None:
    require_amqp_url(options, prefix=_PREFIX)
    exchange = optional_exchange(options, prefix=_PREFIX)
    if exchange is None:
        require_queue(options, prefix=_PREFIX)
    else:
        optional_routing_key(options, prefix=_PREFIX)


def _missing_extra() -> ConfigError:
    return ConfigError(
        'publish.kind = "rabbitmq" requires the pika package; '
        "install with: pip install 'cicerone-recommender[rabbitmq]'"
    )


class RabbitMQPublisher:
    def __init__(self, options: dict[str, Any]):
        validate_rabbitmq_publish_options(options)
        self._amqp_url = require_amqp_url(options, prefix=_PREFIX)
        self._exchange = optional_exchange(options, prefix=_PREFIX) or ""
        self._queue = require_queue(options, prefix=_PREFIX) if not self._exchange else ""
        routing = optional_routing_key(options, prefix=_PREFIX)
        self._routing_key = routing if routing is not None else self._queue
        self._connection: Any | None = None
        self._channel: Any | None = None

    def connect(self) -> None:
        try:
            import pika
        except ImportError as exc:
            raise _missing_extra() from exc
        connection = None
        try:
            connection = pika.BlockingConnection(pika.URLParameters(self._amqp_url))
            channel = connection.channel()
            if self._exchange == "":
                channel.queue_declare(queue=self._queue, durable=True)
        except Exception as exc:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.exception("Failed to close RabbitMQ publisher connection after connect error")
            raise ConfigError(f"publish.options.amqp_url is unreachable: {exc}") from exc
        self._connection = connection
        self._channel = channel

    def publish(self, df: pd.DataFrame) -> None:
        channel = self._require()
        for user_id, body in user_recommendation_messages(df):
            routing_key = self._routing_key or user_id
            channel.basic_publish(exchange=self._exchange, routing_key=routing_key, body=body)

    def close(self) -> None:
        channel = self._channel
        connection = self._connection
        self._channel = None
        self._connection = None
        for handle, label in ((channel, "channel"), (connection, "connection")):
            if handle is None:
                continue
            closer = getattr(handle, "close", None)
            if not callable(closer):
                continue
            try:
                closer()
            except Exception:
                logger.exception("Failed to close RabbitMQ publisher %s", label)

    def _require(self) -> Any:
        if self._channel is None:
            raise RuntimeError("RabbitMQPublisher is not connected")
        return self._channel
