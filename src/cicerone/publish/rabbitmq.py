"""Publish per-user recommendation JSON to a RabbitMQ queue or exchange."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from cicerone.amqp_options import (
    amqp_timeout_seconds,
    apply_amqp_timeouts,
    optional_exchange,
    optional_routing_key,
    require_amqp_url,
    require_queue,
)
from cicerone.config.constants import ConfigError
from cicerone.publish.base import PublishError
from cicerone.publish.payload import user_recommendation_messages

logger = logging.getLogger(__name__)

_PREFIX = "publish.options"


def _rabbitmq_publish_errors() -> tuple[type[BaseException], ...]:
    # Imported lazily: events.errors loads this package while it is initializing.
    from cicerone.events.errors import AMQPError

    return (AMQPError, OSError, TimeoutError)


def _close_quietly(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        # Connect's original error must still propagate.
        logger.exception("Failed to close RabbitMQ publisher connection after connect error")


def validate_rabbitmq_publish_options(options: dict[str, Any]) -> None:
    require_amqp_url(options, prefix=_PREFIX)
    amqp_timeout_seconds(options, prefix=_PREFIX)
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
        try:
            import pika
        except ImportError as exc:
            raise _missing_extra() from exc
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
                _close_quietly(connection)
        self._connection = connection
        self._channel = channel
        self._connected_once = True

    def publish(self, df: pd.DataFrame, *, user_ids: Sequence[str] | None = None) -> None:
        messages = user_recommendation_messages(df, user_ids=user_ids)
        if not messages:
            return
        sent = [0]
        if self._channel is None:
            if self._connected_once:
                self.connect()
            else:
                self._publish_from(messages, sent)
                return
        try:
            self._publish_from(messages, sent)
        except Exception as exc:
            if not isinstance(exc, _rabbitmq_publish_errors()) and not sent[0]:
                raise
            logger.exception("RabbitMQ publish failed; recovering publisher")
            self._retry_unsent(messages, sent)

    def _retry_unsent(self, messages: Sequence[tuple[str, bytes, str]], sent: list[int]) -> None:
        try:
            self._recover()
        except ConfigError as exc:
            self._fail_unless_sent(sent, exc)
            return
        except Exception:
            if not sent[0]:
                raise
            # Accepted records must not be raised back to a caller that would republish them.
            logger.exception("RabbitMQ publisher recover failed after confirm")
            return
        try:
            self._publish_from(messages, sent)
        except Exception as exc:
            if isinstance(exc, _rabbitmq_publish_errors()) or sent[0]:
                self._fail_unless_sent(sent, exc)
                return
            raise

    def _fail_unless_sent(self, sent: list[int], exc: BaseException) -> None:
        if sent[0]:
            logger.exception("RabbitMQ publish stopped after confirm; accepted records are not retried")
            return
        raise PublishError(f"RabbitMQ publish failed: {exc}") from exc

    def _publish_from(self, messages: Sequence[tuple[str, bytes, str]], sent: list[int]) -> None:
        channel = self._require()
        while sent[0] < len(messages):
            _user_id, body, message_id = messages[sent[0]]
            channel.basic_publish(
                exchange=self._exchange,
                routing_key=self._routing_key,
                body=body,
                properties=self._properties(message_id),
            )
            sent[0] += 1

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
        close_exc: BaseException | None = None
        unexpected: BaseException | None = None
        for handle, label in ((channel, "channel"), (connection, "connection")):
            if handle is None:
                continue
            closer = getattr(handle, "close", None)
            if not callable(closer):
                continue
            try:
                closer()
            except Exception as exc:
                logger.exception("Failed to close RabbitMQ publisher %s", label)
                if isinstance(exc, _rabbitmq_publish_errors()):
                    if close_exc is None:
                        close_exc = exc
                    continue
                if unexpected is None:
                    unexpected = exc
        if unexpected is not None:
            raise unexpected
        if close_exc is not None:
            raise PublishError(f"RabbitMQ publisher close failed: {close_exc}") from close_exc

    def _require(self) -> Any:
        if self._channel is None:
            raise PublishError("RabbitMQPublisher is not connected")
        return self._channel
