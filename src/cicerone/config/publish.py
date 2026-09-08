"""``[publish]`` settings coercion and TOML load helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.config.settings import PublishSettings

ResolveEnv = Callable[[Any, str], Any]

_PUBLISH_KINDS = ("kafka", "rabbitmq")


def coerce_publish_settings(value: Any | None) -> PublishSettings:
    if isinstance(value, PublishSettings):
        return PublishSettings(
            enabled=value.enabled,
            kind=value.kind,
            options=dict(value.options),
        )
    if value is None:
        return PublishSettings()
    if not isinstance(value, dict):
        raise TypeError(f"Expected PublishSettings, dict, or None; got {type(value).__name__}")
    raw = dict(value)
    return PublishSettings(
        enabled=bool(raw.get("enabled", False)),
        kind=str(raw.get("kind", "kafka")).lower(),
        options=dict(raw.get("options") or {}),
    )


def load_publish_settings(
    publish_raw: dict[str, Any],
    *,
    resolve_env: ResolveEnv,
) -> PublishSettings:
    enabled = bool(publish_raw.get("enabled", False))
    kind = str(publish_raw.get("kind", "kafka")).lower()
    allowed = _PUBLISH_KINDS
    if enabled and kind not in allowed:
        raise ConfigError(f"publish.kind must be one of {list(allowed)}, got {kind!r}")

    options = resolve_env(publish_raw.get("options", {}), "publish.options")
    if not isinstance(options, dict):
        raise ConfigError("publish.options must be a table")

    if enabled and kind == "kafka":
        from cicerone.publish.kafka import validate_kafka_publish_options

        validate_kafka_publish_options(options)
    if enabled and kind == "rabbitmq":
        from cicerone.publish.rabbitmq import validate_rabbitmq_publish_options

        validate_rabbitmq_publish_options(options)

    return PublishSettings(enabled=enabled, kind=kind, options=options)
