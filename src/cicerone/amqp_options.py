"""Shared AMQP options for events ingest and recs publish."""

from __future__ import annotations

from typing import Any

from cicerone.kafka_options import optional_int, optional_nonempty_str, require_nonempty_str

DEFAULT_PREFETCH = 100


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
