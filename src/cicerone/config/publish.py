"""``[publish]`` settings TOML load helpers."""

from __future__ import annotations

from typing import Any

from cicerone.config.settings import PublishSettings
from cicerone.config.validation import load_kind_section

_PUBLISH_KINDS = ("kafka", "rabbitmq")


def load_publish_settings(publish_raw: dict[str, Any]) -> PublishSettings:
    enabled, kind, options = load_kind_section(publish_raw, "publish", "kafka", _PUBLISH_KINDS)
    if enabled and kind == "kafka":
        from cicerone.publish.kafka import validate_kafka_publish_options

        validate_kafka_publish_options(options)
    if enabled and kind == "rabbitmq":
        from cicerone.publish.rabbitmq import validate_rabbitmq_publish_options

        validate_rabbitmq_publish_options(options)

    return PublishSettings(enabled=enabled, kind=kind, options=options)
