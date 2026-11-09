"""Shared option coercion for broker config (Kafka, AMQP, and similar)."""

from __future__ import annotations

import math
from typing import Any

from cicerone.config.constants import ConfigError

MAX_BROKER_TIMEOUT_SECONDS = (2**31 - 1) / 1000.0  # signed 32-bit milliseconds


# What int()/float() raise on a value that isn't a usable number.
NUMBER_PARSE_ERRORS = (TypeError, ValueError, OverflowError)


def require_nonempty_str(options: dict[str, Any], key: str, *, prefix: str) -> str:
    value = options.get(key)
    if value in (None, "") or (isinstance(value, str) and not str(value).strip()):
        raise ConfigError(f"{prefix}.{key} is required")
    return str(value).strip()


def optional_nonempty_str(options: dict[str, Any], key: str, *, prefix: str) -> str | None:
    if key not in options or options[key] in (None, ""):
        return None
    text = str(options[key]).strip()
    if not text:
        raise ConfigError(f"{prefix}.{key} must be non-empty when set")
    return text


def as_int(options: dict[str, Any], key: str, default: int, *, prefix: str) -> int:
    raw = options.get(key, default)
    try:
        return int(raw)
    except NUMBER_PARSE_ERRORS as exc:
        raise ConfigError(f"{prefix}.{key} must be an integer, got {raw!r}") from exc


def optional_int(options: dict[str, Any], key: str, default: int, *, prefix: str, minimum: int) -> int:
    value = as_int(options, key, default, prefix=prefix)
    if value < minimum:
        raise ConfigError(f"{prefix}.{key} must be >= {minimum}, got {value}")
    return value


def optional_float(
    options: dict[str, Any],
    key: str,
    default: float,
    *,
    prefix: str,
    minimum_exclusive: float = 0.0,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = options.get(key, default)
    try:
        value = float(raw)
    except NUMBER_PARSE_ERRORS as exc:
        raise ConfigError(f"{prefix}.{key} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{prefix}.{key} must be finite, got {raw!r}")
    if value <= minimum_exclusive:
        raise ConfigError(f"{prefix}.{key} must be > {minimum_exclusive}, got {raw!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{prefix}.{key} must be >= {minimum}, got {raw!r}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{prefix}.{key} must be <= {maximum}, got {raw!r}")
    return value
