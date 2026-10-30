"""Broker client errors, with stand-ins when the optional extra is not installed.

A leaf module (no cicerone imports) so publish/ can use it while events.errors is initializing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from confluent_kafka import KafkaException
    from pika.exceptions import AMQPError
    from redis.exceptions import RedisError
else:
    try:
        from confluent_kafka import KafkaException
    except ImportError:

        class KafkaException(Exception):
            pass

    try:
        from pika.exceptions import AMQPError
    except ImportError:

        class AMQPError(Exception):
            pass

    try:
        from redis.exceptions import RedisError
    except ImportError:

        class RedisError(Exception):
            pass


RABBITMQ_IO_ERRORS: tuple[type[BaseException], ...] = (AMQPError, OSError, TimeoutError)
