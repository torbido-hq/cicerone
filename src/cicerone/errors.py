"""Exception groups for optional steps, and the shared log line for caught errors."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from botocore.exceptions import BotoCoreError
from pyarrow.lib import ArrowInvalid, ArrowTypeError
from sqlalchemy.exc import SQLAlchemyError

from cicerone.io.replace_users import RecommendationSchemaError
from cicerone.publish.base import PublishError

logger = logging.getLogger(__name__)

# No bare TypeError: those are bugs and must fail the job. ArrowTypeError is a data/schema mismatch.
OPTIONAL_IO_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    SQLAlchemyError,
    ArrowInvalid,
    ArrowTypeError,
    BotoCoreError,
    RecommendationSchemaError,
)
OPTIONAL_EVAL_ERRORS: tuple[type[BaseException], ...] = (ValueError, LookupError)
PUBLISH_ERRORS: tuple[type[BaseException], ...] = (OSError, ValueError, PublishError)
SINK_WRITE_ERRORS: tuple[type[BaseException], ...] = (*OPTIONAL_IO_ERRORS, RuntimeError)


def log_caught(message: str, exc: BaseException, *, log: logging.Logger | None = None) -> None:
    (log or logger).exception("%s (%s: %s)", message, type(exc).__name__, exc)


T = TypeVar("T")
D = TypeVar("D")


def try_load(
    label: str,
    fn: Callable[[], T],
    default: D,
    *,
    log: logging.Logger,
    errors: tuple[type[BaseException], ...] = OPTIONAL_IO_ERRORS,
) -> T | D:
    try:
        return fn()
    except errors as exc:
        log_caught(f"Failed to {label}", exc, log=log)
        return default
