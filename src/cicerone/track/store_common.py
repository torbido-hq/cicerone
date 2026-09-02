"""Shared track store constants and helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.io.options import storage_backend
from cicerone.track.normalize import assign_missing_event_id
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)

TRACK_FILENAME = "track.jsonl"
EVAL_FILENAME = "track_eval.json"
HISTORY_FILENAME = "recommendation_history.parquet"
HISTORY_DIR = "recommendation_history"
DEFAULT_TRACK_TABLE = "recommendation_track"
DEFAULT_EVAL_TABLE = "recommendation_eval"
DEFAULT_HISTORY_TABLE = "recommendation_history"
TRACK_LOG_BACKEND_ERROR = (
    'track.enabled requires output kind = "db" or a local dataset path; '
    "object-store JSONL append is not atomic"
)
TRACK_LOG_HA_ERROR = 'track.enabled with events.ha requires output kind = "db"'

TRACK_COLUMNS: tuple[str, ...] = (
    "kind",
    USER_COLUMN,
    ITEM_COLUMN,
    RANK_COLUMN,
    "occurred_at",
    "event_id",
    VARIANT_COLUMN,
    "experiment_id",
    "generated_at",
)
HISTORY_COLUMNS: tuple[str, ...] = (
    USER_COLUMN,
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    VARIANT_COLUMN,
    "generated_at",
)


def require_appendable_track_log(output: IOSettings) -> None:
    if output.kind == "db":
        return
    if output.kind == "dataset" and storage_backend(output.options) == "local":
        return
    raise ConfigError(TRACK_LOG_BACKEND_ERROR)


def _row_with_event_id(row: Mapping[str, Any]) -> dict[str, Any]:
    return assign_missing_event_id(row)


def _history_frame(recommendations: pd.DataFrame, generated_at: str) -> pd.DataFrame:
    frame = recommendations.copy()
    frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
    frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
    if RANK_COLUMN in frame.columns:
        frame[RANK_COLUMN] = pd.to_numeric(frame[RANK_COLUMN], errors="coerce")
    else:
        frame[RANK_COLUMN] = pd.NA
    if SOURCE_COLUMN not in frame.columns:
        frame[SOURCE_COLUMN] = None
    if VARIANT_COLUMN not in frame.columns:
        frame[VARIANT_COLUMN] = None
    frame["generated_at"] = generated_at
    return frame.loc[:, list(HISTORY_COLUMNS)]


def _jsonish(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value
