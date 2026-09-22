"""Shared Kafka broker options for events ingest and recs publish."""

from __future__ import annotations

import math
from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.option_parse import (
    MAX_BROKER_TIMEOUT_SECONDS,
    optional_float,
    optional_int,
    optional_nonempty_str,
    require_nonempty_str,
)

SECURITY_PROTOCOLS = frozenset({"plaintext", "ssl", "sasl_plaintext", "sasl_ssl"})
DEFAULT_TIMEOUT_SECONDS = 10.0
MIN_TIMEOUT_MS = 10  # librdkafka socket.timeout.ms
MIN_TIMEOUT_SECONDS = MIN_TIMEOUT_MS / 1000.0
MAX_TIMEOUT_MS = 2**31 - 1
DEFAULT_SESSION_TIMEOUT_MS = 45_000
DEFAULT_MAX_POLL_INTERVAL_MS = 300_000
MIN_SESSION_TIMEOUT_MS = 2


def kafka_timeout_seconds(options: dict[str, Any], *, prefix: str) -> float:
    return optional_float(
        options,
        "timeout_seconds",
        DEFAULT_TIMEOUT_SECONDS,
        prefix=prefix,
        minimum=MIN_TIMEOUT_SECONDS,
        maximum=MAX_BROKER_TIMEOUT_SECONDS,
    )


def kafka_timeout_ms(options: dict[str, Any], *, prefix: str) -> int:
    seconds = kafka_timeout_seconds(options, prefix=prefix)
    try:
        timeout_ms = int(round(seconds * 1000))
    except (OverflowError, ValueError) as exc:
        raise ConfigError(f"{prefix}.timeout_seconds must be a number, got {seconds!r}") from exc
    if timeout_ms < MIN_TIMEOUT_MS or timeout_ms > MAX_TIMEOUT_MS:
        raise ConfigError(
            f"{prefix}.timeout_seconds must be >= {MIN_TIMEOUT_SECONDS} "
            f"and <= {MAX_BROKER_TIMEOUT_SECONDS}, got {seconds!r}"
        )
    return timeout_ms


def kafka_client_config(options: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    timeout_ms = kafka_timeout_ms(options, prefix=prefix)
    conf: dict[str, Any] = {
        "bootstrap.servers": require_nonempty_str(options, "bootstrap_servers", prefix=prefix),
        "socket.timeout.ms": timeout_ms,
        "request.timeout.ms": timeout_ms,
    }
    protocol = optional_nonempty_str(options, "security_protocol", prefix=prefix)
    if protocol is not None:
        key = protocol.lower()
        if key not in SECURITY_PROTOCOLS:
            raise ConfigError(
                f"{prefix}.security_protocol must be one of {sorted(SECURITY_PROTOCOLS)}, got {protocol!r}"
            )
        conf["security.protocol"] = protocol.upper()
    mechanism = optional_nonempty_str(options, "sasl_mechanism", prefix=prefix)
    if mechanism is not None:
        conf["sasl.mechanisms"] = mechanism.upper()
    username = optional_nonempty_str(options, "sasl_username", prefix=prefix)
    if username is not None:
        conf["sasl.username"] = username
    password = optional_nonempty_str(options, "sasl_password", prefix=prefix)
    if password is not None:
        conf["sasl.password"] = password
    return conf


def kafka_consumer_config(options: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    conf = kafka_client_config(options, prefix=prefix)
    max_poll = _optional_timeout_ms(options, "max_poll_interval_ms", prefix=prefix)
    session = _optional_timeout_ms(
        options, "session_timeout_ms", prefix=prefix, minimum=MIN_SESSION_TIMEOUT_MS
    )
    effective_max_poll = max_poll if max_poll is not None else DEFAULT_MAX_POLL_INTERVAL_MS
    effective_session = session if session is not None else DEFAULT_SESSION_TIMEOUT_MS
    if effective_max_poll < effective_session:
        max_name = f"{prefix}.max_poll_interval_ms"
        session_name = f"{prefix}.session_timeout_ms"
        if max_poll is None:
            max_name = f"{max_name} (librdkafka default {DEFAULT_MAX_POLL_INTERVAL_MS})"
        if session is None:
            session_name = f"{session_name} (librdkafka default {DEFAULT_SESSION_TIMEOUT_MS})"
        raise ConfigError(
            f"{max_name} must be >= {session_name}, got {effective_max_poll} < {effective_session}"
        )
    if max_poll is not None:
        conf["max.poll.interval.ms"] = max_poll
    if session is not None:
        conf["session.timeout.ms"] = session
        conf["heartbeat.interval.ms"] = _heartbeat_interval_ms(session)
    return conf


def _heartbeat_interval_ms(session: int) -> int:
    heartbeat = max(1, min(session - 1, session // 3))
    if heartbeat >= session:
        raise ConfigError(
            f"session_timeout_ms must be >= {MIN_SESSION_TIMEOUT_MS} so "
            f"heartbeat.interval.ms stays below session.timeout.ms, got {session}"
        )
    return heartbeat


def _optional_timeout_ms(options: dict[str, Any], key: str, *, prefix: str, minimum: int = 1) -> int | None:
    if key not in options or options[key] in (None, ""):
        return None
    raw = options[key]
    if isinstance(raw, bool) or (isinstance(raw, float) and (not math.isfinite(raw) or not raw.is_integer())):
        raise ConfigError(f"{prefix}.{key} must be an integer, got {raw!r}")
    value = optional_int(options, key, 0, prefix=prefix, minimum=minimum)
    if value > MAX_TIMEOUT_MS:
        raise ConfigError(f"{prefix}.{key} must be <= {MAX_TIMEOUT_MS}, got {value}")
    return value
