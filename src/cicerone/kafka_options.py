"""Shared Kafka broker options for events ingest and recs publish."""

from __future__ import annotations

from typing import Any

from cicerone.config.constants import ConfigError

SECURITY_PROTOCOLS = frozenset({"plaintext", "ssl", "sasl_plaintext", "sasl_ssl"})


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


def optional_int(options: dict[str, Any], key: str, default: int, *, prefix: str, minimum: int) -> int:
    raw = options.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{prefix}.{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{prefix}.{key} must be >= {minimum}, got {value}")
    return value


def kafka_client_config(options: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    conf: dict[str, Any] = {
        "bootstrap.servers": require_nonempty_str(options, "bootstrap_servers", prefix=prefix),
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
