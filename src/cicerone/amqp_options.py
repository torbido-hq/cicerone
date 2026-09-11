"""Shared AMQP options for events ingest and recs publish."""

from __future__ import annotations

from typing import Any

from cicerone.option_parse import (
    MAX_BROKER_TIMEOUT_SECONDS,
    optional_float,
    optional_int,
    optional_nonempty_str,
    require_nonempty_str,
)

DEFAULT_PREFETCH = 100
DEFAULT_TIMEOUT_SECONDS = 10.0


def amqp_timeout_seconds(options: dict[str, Any], *, prefix: str) -> float:
    return optional_float(
        options,
        "timeout_seconds",
        DEFAULT_TIMEOUT_SECONDS,
        prefix=prefix,
        maximum=MAX_BROKER_TIMEOUT_SECONDS,
    )


def apply_amqp_timeouts(params: Any, timeout_seconds: float) -> Any:
    params.socket_timeout = timeout_seconds
    params.blocked_connection_timeout = timeout_seconds
    params.stack_timeout = timeout_seconds
    return params


def require_amqp_url(options: dict[str, Any], *, prefix: str) -> str:
    return require_nonempty_str(options, "amqp_url", prefix=prefix)


def require_queue(options: dict[str, Any], *, prefix: str) -> str:
    return require_nonempty_str(options, "queue", prefix=prefix)


def prefetch_count(options: dict[str, Any], *, prefix: str) -> int:
    return optional_int(options, "prefetch", DEFAULT_PREFETCH, prefix=prefix, minimum=1)


def optional_exchange(options: dict[str, Any], *, prefix: str) -> str | None:
    return optional_nonempty_str(options, "exchange", prefix=prefix)


def optional_routing_key(options: dict[str, Any], *, prefix: str) -> str | None:
    return optional_nonempty_str(options, "routing_key", prefix=prefix)
