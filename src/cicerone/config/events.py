"""``[events]`` settings coercion and TOML load helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cicerone.config.constants import (
    DEFAULT_EVENTS_BATCH_SIZE,
    DEFAULT_EVENTS_BATCH_WINDOW_SECONDS,
    EVENT_SOURCE_KINDS,
    ConfigError,
)
from cicerone.config.settings import EventsIncrementalSettings, EventsSettings
from cicerone.config.validation import require_positive_float, require_positive_int

ResolveEnv = Callable[[Any, str], Any]


def coerce_events_settings(value: Any | None) -> EventsSettings:
    if isinstance(value, EventsSettings):
        return value
    if value is None:
        return EventsSettings()
    if not isinstance(value, dict):
        raise TypeError(f"Expected EventsSettings, dict, or None; got {type(value).__name__}")
    raw = dict(value)
    incremental_raw = raw.pop("incremental", None)
    if isinstance(incremental_raw, EventsIncrementalSettings):
        incremental = incremental_raw
    elif isinstance(incremental_raw, dict):
        incremental = EventsIncrementalSettings(**incremental_raw)
    elif incremental_raw is None:
        incremental = EventsIncrementalSettings()
    else:
        raise TypeError(
            f"Expected EventsIncrementalSettings, dict, or None; got {type(incremental_raw).__name__}"
        )
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
    if enabled and kind not in EVENT_SOURCE_KINDS:
        raise ConfigError(f"events.kind must be one of {list(EVENT_SOURCE_KINDS)}, got {kind!r}")

    options = resolve_env(events_raw.get("options", {}), "events.options")
    if not isinstance(options, dict):
        raise ConfigError("events.options must be a table")
    auth_token = (
        resolve_env(options["auth_token"], "events.options.auth_token") if "auth_token" in options else None
    )
    if auth_token is not None:
        options = {**options, "auth_token": auth_token}

    incremental_raw = events_raw.get("incremental", {}) or {}
    batch_size = require_positive_int(
        int(incremental_raw.get("batch_size", DEFAULT_EVENTS_BATCH_SIZE)),
        name="events.incremental.batch_size",
    )
    batch_window_seconds = require_positive_float(
        float(incremental_raw.get("batch_window_seconds", DEFAULT_EVENTS_BATCH_WINDOW_SECONDS)),
        name="events.incremental.batch_window_seconds",
    )

    if enabled and kind == "webhook" and mode == "serve":
        webhook_token = auth_token or serve_auth_token
        if not webhook_token:
            raise ConfigError(
                "events.options.auth_token or serve.auth_token is required when "
                'events.enabled = true, events.kind = "webhook", and job.mode = "serve"'
            )

    return EventsSettings(
        enabled=enabled,
        kind=kind,
        options=options,
        incremental=EventsIncrementalSettings(
            batch_size=batch_size,
            batch_window_seconds=batch_window_seconds,
        ),
    )
