"""``[track]`` and ``job.eval`` settings TOML load helpers."""

from __future__ import annotations

from typing import Any

from cicerone.config.constants import (
    DEFAULT_TRACK_ATTRIBUTION_WINDOW_HOURS,
    DEFAULT_TRACK_MIN_IMPRESSIONS,
    ConfigError,
)
from cicerone.config.settings import EvalSettings, TrackSettings
from cicerone.config.validation import require_non_negative_int, require_positive_int


def _load_string_tuple(raw: object, *, name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise ConfigError(f"{name} must be a list of non-empty strings")
    return tuple(item.strip() for item in raw)


def _load_positive_int_tuple(raw: object, *, name: str) -> tuple[int, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(f"{name} must be a list of positive integers")
    values: list[int] = []
    for item in raw:
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be a list of positive integers") from exc
        values.append(require_positive_int(number, name=name))
    return tuple(values)


def load_track_settings(raw: dict[str, Any] | None) -> TrackSettings:
    data = raw or {}
    enabled = bool(data.get("enabled", False))
    window = float(data.get("attribution_window_hours", DEFAULT_TRACK_ATTRIBUTION_WINDOW_HOURS))
    if window <= 0:
        raise ConfigError(f"track.attribution_window_hours must be > 0, got {window}")
    min_impressions = require_non_negative_int(
        int(data.get("min_impressions", DEFAULT_TRACK_MIN_IMPRESSIONS)),
        name="track.min_impressions",
    )
    return TrackSettings(
        enabled=enabled,
        attribution_window_hours=window,
        conversion_event_types=_load_string_tuple(
            data.get("conversion_event_types"), name="track.conversion_event_types"
        ),
        min_impressions=min_impressions,
    )


def load_eval_settings(raw: dict[str, Any] | None) -> EvalSettings:
    data = raw or {}
    ks_raw = data.get("ks")
    ks = _load_positive_int_tuple(ks_raw, name="job.eval.ks") if ks_raw is not None else ()
    return EvalSettings(
        enabled=bool(data.get("enabled", False)),
        event_types=_load_string_tuple(data.get("event_types"), name="job.eval.event_types"),
        ks=ks,
    )
