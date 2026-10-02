"""Erase one user's recommendations, track rows, exposures, and history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.experiment.store import EXPOSURES_FILENAME, ExperimentStore
from cicerone.io.db_errors import is_missing_column_error
from cicerone.io.db_store import DEFAULT_RECOMMENDATIONS_TABLE, MISSING_TABLE_ERRORS
from cicerone.io.factory import build_output_sink
from cicerone.io.jsonl_user import drop_user_lines
from cicerone.io.options import read_parquet, require_option, sql_identifier, storage_backend
from cicerone.io.recommendation_schema import USER_COLUMN, recommendations_sql_names
from cicerone.locks.hold import LockBackend
from cicerone.track.store import TrackStore
from cicerone.track.store_common import HISTORY_DIR, HISTORY_FILENAME, TRACK_FILENAME

FORGET_OUTPUT_ERROR = 'forget-user requires output kind = "db" or a local dataset path'


@dataclass(frozen=True)
class ForgetResult:
    """Rows removed for one user. Missing files and tables count as zero."""

    recommendations: int
    track: int
    exposures: int
    history: int


def require_forget_output(output: IOSettings) -> None:
    """Reject object-store output. JSONL rewrite there is not atomic."""
    if output.kind == "db":
        return
    if output.kind == "dataset" and storage_backend(output.options) == "local":
        return
    raise ConfigError(FORGET_OUTPUT_ERROR)


def forget_user(
    output: IOSettings,
    user_id: str,
    *,
    writer_lock: LockBackend | None = None,
) -> ForgetResult:
    """Drop one user's output rows. Catalog users and events are left in place."""
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id is required")
    require_forget_output(output)
    if output.kind == "dataset":
        _preflight_local(output)
    recommendations = _delete_recommendations(output, user_id, writer_lock=writer_lock)
    track_store = TrackStore(output, writer_lock=writer_lock)
    track = track_store.delete_user_rows(user_id)
    exposures = ExperimentStore(output, writer_lock=writer_lock).delete_exposures_for_user(user_id)
    history = track_store.delete_history_for_user(user_id)
    return ForgetResult(
        recommendations=recommendations,
        track=track,
        exposures=exposures,
        history=history,
    )


def _preflight_local(output: IOSettings) -> None:
    root = Path(require_option(output.options, "path", "local"))
    for name in (TRACK_FILENAME, EXPOSURES_FILENAME):
        path = root / name
        if path.is_file():
            drop_user_lines(path.read_bytes(), user_id="")
    legacy = root / HISTORY_FILENAME
    if legacy.is_file():
        pd.read_parquet(legacy)
    history = root / HISTORY_DIR
    if not history.is_dir():
        return
    for path in sorted(history.glob("*.parquet")):
        pd.read_parquet(path)


def _delete_recommendations(
    output: IOSettings,
    user_id: str,
    *,
    writer_lock: LockBackend | None,
) -> int:
    removed = _count_recommendation_rows(output, user_id)
    if not removed:
        return 0
    build_output_sink(output, writer_lock=writer_lock).replace_recommendations_for_users(
        pd.DataFrame(),
        user_ids=[user_id],
    )
    return removed


def _count_recommendation_rows(output: IOSettings, user_id: str) -> int:
    if output.kind == "db":
        return _count_db_recommendations(output, user_id)
    try:
        frame = read_parquet(output.options, "recommendations.parquet")
    except FileNotFoundError:
        return 0
    if frame.empty or USER_COLUMN not in frame.columns:
        return 0
    return int((frame[USER_COLUMN].astype(str) == user_id).sum())


def _count_db_recommendations(output: IOSettings, user_id: str) -> int:
    table, _columns, user_col = recommendations_sql_names(
        output.options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
    )
    table = sql_identifier(table, option="recommendations_table")
    user_col = sql_identifier(user_col, option="recommendations_column")
    engine = create_engine(require_option(output.options, "database_url", "db"), pool_pre_ping=True)
    try:
        return _count_where(engine, table, user_col, user_id)
    finally:
        engine.dispose()


def _count_where(engine: Engine, table: str, user_col: str, user_id: str) -> int:
    statement = text(f'SELECT COUNT(*) FROM "{table}" WHERE "{user_col}" = :user_id')
    try:
        with engine.connect() as conn:
            value = conn.execute(statement, {"user_id": user_id}).scalar()
    except MISSING_TABLE_ERRORS as exc:
        if is_missing_column_error(exc):
            raise
        return 0
    return int(value or 0)
