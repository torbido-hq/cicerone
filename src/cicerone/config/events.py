"""``[events]`` settings coercion and TOML load helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.config.constants import (
    DEFAULT_EVENTS_BATCH_SIZE,
    DEFAULT_EVENTS_BATCH_WINDOW_SECONDS,
    DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
    ConfigError,
)
from cicerone.config.settings import EventsIncrementalSettings, EventsSettings
from cicerone.config.validation import require_positive_float, require_positive_int
from cicerone.events.registry import registered_event_source_kinds

ResolveEnv = Callable[[Any, str], Any]


def _incremental_settings(
    raw: dict[str, Any] | EventsIncrementalSettings | None,
) -> EventsIncrementalSettings:
    if isinstance(raw, EventsIncrementalSettings):
        data = {
            "batch_size": raw.batch_size,
            "batch_window_seconds": raw.batch_window_seconds,
            "poll_interval_seconds": raw.poll_interval_seconds,
        }
    elif raw is None:
        data = {}
    elif isinstance(raw, dict):
        data = raw
    else:
        raise TypeError(f"Expected EventsIncrementalSettings, dict, or None; got {type(raw).__name__}")
    return EventsIncrementalSettings(
        batch_size=require_positive_int(
            int(data.get("batch_size", DEFAULT_EVENTS_BATCH_SIZE)),
            name="events.incremental.batch_size",
        ),
        batch_window_seconds=require_positive_float(
            float(data.get("batch_window_seconds", DEFAULT_EVENTS_BATCH_WINDOW_SECONDS)),
            name="events.incremental.batch_window_seconds",
        ),
        poll_interval_seconds=require_positive_float(
            float(data.get("poll_interval_seconds", DEFAULT_EVENTS_POLL_INTERVAL_SECONDS)),
            name="events.incremental.poll_interval_seconds",
        ),
    )


def coerce_events_settings(value: Any | None) -> EventsSettings:
    if isinstance(value, EventsSettings):
        return EventsSettings(
            enabled=value.enabled,
            kind=value.kind,
            options=dict(value.options),
            incremental=_incremental_settings(value.incremental),
            ha=value.ha,
        )
    if value is None:
        return EventsSettings()
    if not isinstance(value, dict):
        raise TypeError(f"Expected EventsSettings, dict, or None; got {type(value).__name__}")
    raw = dict(value)
    incremental = _incremental_settings(raw.pop("incremental", None))
    return EventsSettings(incremental=incremental, **raw)


def load_events_settings(
    events_raw: dict[str, Any],
    *,
    mode: str,
    serve_auth_token: str | None,
    resolve_env: ResolveEnv,
) -> EventsSettings:
    """Parse ``[events]`` from TOML; ``resolve_env`` is ``load._resolve_env_placeholders``."""
    enabled = bool(events_raw.get("enabled", False))
    kind = str(events_raw.get("kind", "webhook")).lower()
    allowed_kinds = registered_event_source_kinds()
    if enabled and kind not in allowed_kinds:
        raise ConfigError(f"events.kind must be one of {list(allowed_kinds)}, got {kind!r}")

    options = resolve_env(events_raw.get("options", {}), "events.options")
    if not isinstance(options, dict):
        raise ConfigError("events.options must be a table")
    auth_token = (
        resolve_env(options["auth_token"], "events.options.auth_token") if "auth_token" in options else None
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
    ha = bool(events_raw.get("ha", False))

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

    return EventsSettings(
        enabled=enabled,
        kind=kind,
        options=options,
        incremental=incremental,
        ha=ha,
    )
