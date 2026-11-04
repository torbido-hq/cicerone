"""Publish per-user recommendation JSON to a RabbitMQ queue or exchange."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from cicerone.amqp_options import (
    amqp_timeout_seconds,
    apply_amqp_timeouts,
    close_amqp_handle,
    optional_exchange,
    optional_routing_key,
    require_amqp_url,
    require_queue,
)
from cicerone.config.constants import ConfigError, import_extra
from cicerone.publish.base import PublishError
from cicerone.publish.payload import user_recommendation_messages

logger = logging.getLogger(__name__)

_PREFIX = "publish.options"


@dataclass
class _Progress:
    sent: int = 0


def _rabbitmq_publish_errors() -> tuple[type[BaseException], ...]:
    # Imported lazily: events.errors loads this package while it is initializing.
    from cicerone.events.errors import AMQPError

    return (AMQPError, OSError, TimeoutError)


def validate_rabbitmq_publish_options(options: dict[str, Any]) -> None:
    require_amqp_url(options, prefix=_PREFIX)
    amqp_timeout_seconds(options, prefix=_PREFIX)
    exchange = optional_exchange(options, prefix=_PREFIX)
    if exchange is None:
        require_queue(options, prefix=_PREFIX)
    else:
        optional_routing_key(options, prefix=_PREFIX)


class RabbitMQPublisher:
    def __init__(self, options: dict[str, Any]):
        validate_rabbitmq_publish_options(options)
        self._amqp_url = require_amqp_url(options, prefix=_PREFIX)
        self._timeout_seconds = amqp_timeout_seconds(options, prefix=_PREFIX)
        self._exchange = optional_exchange(options, prefix=_PREFIX) or ""
        self._queue = require_queue(options, prefix=_PREFIX) if not self._exchange else ""
        routing = optional_routing_key(options, prefix=_PREFIX)
        self._routing_key = routing if routing is not None else self._queue
        self._connection: Any | None = None
        self._channel: Any | None = None
        self._connected_once = False

    def connect(self) -> None:
        if self._channel is not None:
            return
        pika = import_extra("pika", "rabbitmq", 'publish.kind = "rabbitmq"')
        connection = None
        connected = False
        try:
            connection = pika.BlockingConnection(
                apply_amqp_timeouts(pika.URLParameters(self._amqp_url), self._timeout_seconds)
            )
            channel = connection.channel()
            confirm = getattr(channel, "confirm_delivery", None)
            if callable(confirm):
                confirm()
            if self._exchange == "":
                channel.queue_declare(queue=self._queue, durable=True)
            connected = True
        except _rabbitmq_publish_errors() as exc:
            raise ConfigError(f"publish.options.amqp_url is unreachable or setup failed: {exc}") from exc
        finally:
            if connection is not None and not connected:
                close_amqp_handle(connection, "publisher connection after connect error", logger)
        self._connection = connection
        self._channel = channel
        self._connected_once = True

    def publish(self, df: pd.DataFrame, *, user_ids: Sequence[str] | None = None) -> None:
        messages = user_recommendation_messages(df, user_ids=user_ids)
        if not messages:
            return
        progress = _Progress()
        if self._channel is None:
            if self._connected_once:
                self.connect()
            else:
                self._publish_from(messages, progress)
                return
        try:
            self._publish_from(messages, progress)
        except Exception as exc:
            if not isinstance(exc, _rabbitmq_publish_errors()) and not progress.sent:
                raise
            logger.exception("RabbitMQ publish failed; recovering publisher")
            self._retry_unsent(messages, progress)

    def _retry_unsent(self, messages: Sequence[tuple[str, bytes, str]], progress: _Progress) -> None:
        try:
            self._recover()
            self._publish_from(messages, progress)
        except Exception as exc:
            if not isinstance(exc, (ConfigError, *_rabbitmq_publish_errors())) and not progress.sent:
                raise
            # Callers republish every user; consumers dedupe on the stable message_id.
            raise PublishError(f"RabbitMQ publish failed: {exc}") from exc

    def _publish_from(self, messages: Sequence[tuple[str, bytes, str]], progress: _Progress) -> None:
        channel = self._require()
        while progress.sent < len(messages):
            _user_id, body, message_id = messages[progress.sent]
            channel.basic_publish(
                exchange=self._exchange,
                routing_key=self._routing_key,
                body=body,
                properties=self._properties(message_id),
            )
            progress.sent += 1

    def _properties(self, message_id: str) -> Any:
        try:
            import pika
        except ImportError:
            return None
        return pika.BasicProperties(message_id=message_id, content_type="application/json", delivery_mode=2)

    def _recover(self) -> None:
        try:
            self.close()
        except PublishError:
            logger.exception("RabbitMQ publisher close failed during recover")
        self.connect()

    def close(self) -> None:
        channel = self._channel
        connection = self._connection
        self._channel = None
        self._connection = None
        errors = [
            exc
            for exc in (
                close_amqp_handle(channel, "publisher channel", logger),
                close_amqp_handle(connection, "publisher connection", logger),
            )
            if exc is not None
        ]
        unexpected = [exc for exc in errors if not isinstance(exc, _rabbitmq_publish_errors())]
        if unexpected:
            raise unexpected[0]
        if errors:
            raise PublishError(f"RabbitMQ publisher close failed: {errors[0]}") from errors[0]

    def _require(self) -> Any:
        if self._channel is None:
            raise PublishError("RabbitMQPublisher is not connected")
        return self._channel
