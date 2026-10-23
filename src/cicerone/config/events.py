"""``[events]`` settings TOML load helpers."""

from __future__ import annotations

from typing import Any

from cicerone.config.constants import (
    DEFAULT_EVENTS_BATCH_SIZE,
    DEFAULT_EVENTS_BATCH_WINDOW_SECONDS,
    DEFAULT_EVENTS_ONLINE_FIT_MIN_EVENTS,
    DEFAULT_EVENTS_ONLINE_FIT_PARTIAL_EPOCHS,
    DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS,
    DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
    ConfigError,
)
from cicerone.config.settings import EventsIncrementalSettings, EventsOnlineSettings, EventsSettings
from cicerone.config.validation import (
    float_field,
    int_field,
    load_kind_section,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
    resolve_env_placeholders,
)
from cicerone.events.registry import registered_event_source_kinds


def _incremental_settings(data: dict[str, Any]) -> EventsIncrementalSettings:
    return EventsIncrementalSettings(
        batch_size=int_field(
            data, "events.incremental.batch_size", DEFAULT_EVENTS_BATCH_SIZE, require_positive_int
        ),
        batch_window_seconds=float_field(
            data,
            "events.incremental.batch_window_seconds",
            DEFAULT_EVENTS_BATCH_WINDOW_SECONDS,
            require_positive_float,
        ),
        poll_interval_seconds=float_field(
            data,
            "events.incremental.poll_interval_seconds",
            DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
            require_positive_float,
        ),
    )


def _online_settings(data: dict[str, Any]) -> EventsOnlineSettings:
    return EventsOnlineSettings(
        enabled=bool(data.get("enabled", False)),
        fit_partial_epochs=int_field(
            data,
            "events.online.fit_partial_epochs",
            DEFAULT_EVENTS_ONLINE_FIT_PARTIAL_EPOCHS,
            require_non_negative_int,
        ),
        fit_min_events=int_field(
            data, "events.online.fit_min_events", DEFAULT_EVENTS_ONLINE_FIT_MIN_EVENTS, require_positive_int
        ),
        max_extra_interactions=int_field(
            data,
            "events.online.max_extra_interactions",
            DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS,
            require_positive_int,
        ),
    )


def load_events_settings(
    events_raw: dict[str, Any],
    *,
    mode: str,
    serve_auth_token: str | None,
) -> EventsSettings:
    enabled, kind, options = load_kind_section(
        events_raw, "events", "webhook", registered_event_source_kinds()
    )
    auth_token = (
        resolve_env_placeholders(options["auth_token"], "events.options.auth_token")
        if "auth_token" in options
        else None
    )
    if auth_token is not None:
        options = {**options, "auth_token": auth_token}

    if "incremental" in events_raw:
        incremental_raw = events_raw["incremental"]
        if not isinstance(incremental_raw, dict):
            raise ConfigError("events.incremental must be a table")
    else:
        incremental_raw = {}
    incremental = _incremental_settings(incremental_raw)
    if "online" in events_raw:
        online_raw = events_raw["online"]
        if not isinstance(online_raw, dict):
            raise ConfigError("events.online must be a table")
    else:
        online_raw = {}
    online = _online_settings(online_raw)
    ha = bool(events_raw.get("ha", False))
    if online.enabled and not enabled:
        raise ConfigError("events.online.enabled requires events.enabled = true")

    if enabled and kind == "webhook" and mode == "serve":
        webhook_token = auth_token or serve_auth_token
        if not webhook_token:
            raise ConfigError(
                "events.options.auth_token or serve.auth_token is required when "
                'events.enabled = true, events.kind = "webhook", and job.mode = "serve"'
            )
    if enabled and kind == "db" and not options.get("database_url"):
        raise ConfigError('events.options.database_url is required when events.kind = "db"')
    if enabled and kind == "s3":
        from cicerone.events.s3 import validate_s3_event_options

        validate_s3_event_options(options)
    if enabled and kind == "redis_streams":
        from cicerone.events.redis_streams import validate_redis_stream_options

        validate_redis_stream_options(options)
    if enabled and kind == "kafka":
        from cicerone.events.kafka import validate_kafka_event_options

        validate_kafka_event_options(options)
    if enabled and kind == "rabbitmq":
        from cicerone.events.rabbitmq import validate_rabbitmq_event_options

        validate_rabbitmq_event_options(options)

    return EventsSettings(
        enabled=enabled,
        kind=kind,
        options=options,
        incremental=incremental,
        ha=ha,
        online=online,
    )
