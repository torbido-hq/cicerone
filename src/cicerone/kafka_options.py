"""Shared Kafka broker options for events ingest and recs publish."""

from __future__ import annotations

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
    session = _optional_timeout_ms(options, "session_timeout_ms", prefix=prefix)
    if max_poll is not None and session is not None and max_poll < session:
        raise ConfigError(
            f"{prefix}.max_poll_interval_ms must be >= {prefix}.session_timeout_ms, "
            f"got {max_poll} < {session}"
        )
    if max_poll is not None:
        conf["max.poll.interval.ms"] = max_poll
    if session is not None:
        conf["session.timeout.ms"] = session
        # librdkafka default heartbeat is 3000ms; session must be larger.
        conf["heartbeat.interval.ms"] = max(1, min(session - 1, session // 3))
    return conf


def _optional_timeout_ms(options: dict[str, Any], key: str, *, prefix: str) -> int | None:
    if key not in options or options[key] in (None, ""):
        return None
    raw = options[key]
    if isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer()):
        raise ConfigError(f"{prefix}.{key} must be an integer, got {raw!r}")
    try:
        value = optional_int(options, key, 0, prefix=prefix, minimum=1)
    except OverflowError as exc:
        raise ConfigError(f"{prefix}.{key} must be an integer, got {raw!r}") from exc
    if value > MAX_TIMEOUT_MS:
        raise ConfigError(f"{prefix}.{key} must be <= {MAX_TIMEOUT_MS}, got {value}")
    return value
