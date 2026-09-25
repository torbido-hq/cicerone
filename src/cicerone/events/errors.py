"""Named errors for event-source I/O and incremental apply."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError

from cicerone.events.base import EventBackpressureError, EventSourceError
from cicerone.io.blob import S3_READ_ERRORS
from cicerone.job_eval import PUBLISH_ERRORS, SINK_WRITE_ERRORS

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


EVENT_SOURCE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    TimeoutError,
    EventBackpressureError,
    EventSourceError,
    SQLAlchemyError,
    KafkaException,
    AMQPError,
    RedisError,
    *S3_READ_ERRORS,
)
_BROAD_PROGRAMMING_ERRORS = (RuntimeError, ValueError, TypeError)
EVENT_APPLY_ERRORS: tuple[type[BaseException], ...] = tuple(
    error
    for error in (*SINK_WRITE_ERRORS, *PUBLISH_ERRORS, *S3_READ_ERRORS)
    if error not in _BROAD_PROGRAMMING_ERRORS
)
EVENT_WORKER_ERRORS: tuple[type[BaseException], ...] = (*EVENT_SOURCE_ERRORS, *EVENT_APPLY_ERRORS)
