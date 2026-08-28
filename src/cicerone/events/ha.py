"""Helpers for incremental-events horizontal HA (leader-only apply)."""

from __future__ import annotations

from typing import Any


def ingest_is_fanout(kind: str, options: dict[str, Any] | None = None) -> bool:
    """True when multiple replicas may safely claim ingest (apply still locked)."""
    if kind in {"redis_streams", "kafka", "rabbitmq"}:
        return True
    if kind != "s3":
        return False
    options = options or {}
    mode = options.get("mode")
    if mode is None:
        return bool(options.get("queue_url"))
    return str(mode).lower() == "sqs"


def poll_without_apply_lock(kind: str, options: dict[str, Any] | None = None) -> bool:
    """Webhook (local queue) and fan-out sources may poll without the apply lease."""
    return kind == "webhook" or ingest_is_fanout(kind, options)
