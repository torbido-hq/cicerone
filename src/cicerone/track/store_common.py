"""Shared track store constants and helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.io.options import storage_backend
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)
from cicerone.track.normalize import assign_missing_event_id

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


def _iso_utc(value: str | None) -> str | None:
    if not value:
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return value
    return stamp.isoformat()


def _stamp_before(value: str, since: str) -> bool:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    start = pd.to_datetime(since, utc=True, errors="coerce")
    if pd.isna(stamp) or pd.isna(start):
        return False
    return bool(stamp < start)


_HISTORY_STEM = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}T)"
    r"(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})"
    r"(?P<frac>\.\d+)?"
    r"(?P<tz>Z|[+-]\d{2}-\d{2})?$"
)


def _unslug_history_stem(stem: str) -> str:
    match = _HISTORY_STEM.fullmatch(stem)
    if not match:
        return stem
    restored = f"{match['date']}{match['h']}:{match['m']}:{match['s']}{match['frac'] or ''}"
    tz = match["tz"]
    if tz and tz != "Z":
        return f"{restored}{tz[:3]}:{tz[4:]}"
    return f"{restored}{tz or ''}"


def _history_stem_before(stem: str, since: str) -> bool:
    return _stamp_before(_unslug_history_stem(stem), since)


def _row_matches(
    row: Mapping[str, Any],
    *,
    kind: str | None,
    experiment_id: str | None,
    since: str | None,
) -> bool:
    if kind is not None and str(row.get("kind") or "") != kind:
        return False
    if experiment_id:
        row_id = str(row.get("experiment_id") or "")
        if row_id and row_id != experiment_id:
            return False
    occurred = str(row.get("occurred_at") or "")
    return not (since and occurred and _stamp_before(occurred, since))


def _track_row_sql_filter(
    *,
    kind: str | None,
    experiment_id: str | None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if kind is not None:
        clauses.append("kind = :kind")
        params["kind"] = kind
    if experiment_id:
        clauses.append("(experiment_id IS NULL OR experiment_id = '' OR experiment_id = :experiment_id)")
        params["experiment_id"] = experiment_id
    if not clauses:
        return "", {}
    return " WHERE " + " AND ".join(clauses), params


def _history_sql_filter(
    *,
    generated_ats: set[str] | None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if generated_ats:
        clauses.append("generated_at IN :generated_ats")
        params["generated_ats"] = sorted(generated_ats)
    if not clauses:
        return "", {}
    return " WHERE " + " AND ".join(clauses), params


def _filter_history(
    frame: pd.DataFrame,
    *,
    generated_ats: set[str] | None,
    since: str | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "generated_at" not in frame.columns:
        if generated_ats is not None or since:
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        return frame
    keep = pd.Series(True, index=frame.index)
    if generated_ats is not None:
        keep &= frame["generated_at"].astype(str).isin(generated_ats)
    if since:
        stamps = pd.to_datetime(frame["generated_at"], utc=True, errors="coerce")
        start = pd.to_datetime(since, utc=True, errors="coerce")
        if pd.notna(start):
            keep &= stamps >= start
    return frame.loc[keep].reset_index(drop=True)
