"""Named errors for event-source I/O and incremental apply."""

from __future__ import annotations

from pandas.errors import DatabaseError
from pyarrow.lib import ArrowException
from sqlalchemy.exc import SQLAlchemyError

from cicerone.broker_errors import AMQPError, KafkaException, RedisError
from cicerone.events.base import EventBackpressureError, EventSourceError
from cicerone.io.blob import S3_READ_ERRORS
from cicerone.io.replace_users import RecommendationSchemaError
from cicerone.publish.base import PublishError

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
EVENT_APPLY_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    SQLAlchemyError,
    DatabaseError,
    ArrowException,
    RecommendationSchemaError,
    PublishError,
    *S3_READ_ERRORS,
)
EVENT_WORKER_ERRORS: tuple[type[BaseException], ...] = (*EVENT_SOURCE_ERRORS, *EVENT_APPLY_ERRORS)
