"""Shared Kafka broker options for events ingest and recs publish."""

from __future__ import annotations

from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.option_parse import optional_nonempty_str, require_nonempty_str

SECURITY_PROTOCOLS = frozenset({"plaintext", "ssl", "sasl_plaintext", "sasl_ssl"})


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
