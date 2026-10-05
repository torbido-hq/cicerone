"""Erase one user's recommendations, track rows, exposures, and history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.experiment.store import EXPOSURES_FILENAME, ExperimentStore
from cicerone.io.db_store import DEFAULT_RECOMMENDATIONS_TABLE
from cicerone.io.factory import build_output_sink
from cicerone.io.jsonl_user import drop_user_lines
from cicerone.io.options import read_parquet, require_option, storage_backend
from cicerone.io.recommendation_schema import USER_COLUMN, recommendations_sql_names
from cicerone.locks.hold import LockBackend
from cicerone.track.store import TrackStore
from cicerone.track.store_common import HISTORY_DIR, HISTORY_FILENAME, TRACK_FILENAME
from cicerone.track.store_db import delete_rows_for_user

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
    if output.kind == "db":
        return _forget_db(output, user_id, writer_lock=writer_lock)
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
    recommendations = root / "recommendations.parquet"
    if recommendations.is_file():
        _require_user_column(recommendations, pd.read_parquet(recommendations))
    for name in (TRACK_FILENAME, EXPOSURES_FILENAME):
        path = root / name
        if path.is_file():
            drop_user_lines(path.read_bytes(), user_id="")
    legacy = root / HISTORY_FILENAME
    if legacy.is_file():
        _require_user_column(legacy, pd.read_parquet(legacy))
    history = root / HISTORY_DIR
    if not history.is_dir():
        return
    for path in sorted(history.glob("*.parquet")):
        _require_user_column(path, pd.read_parquet(path))


def _require_user_column(path: Path, frame: pd.DataFrame) -> None:
    if not frame.empty and USER_COLUMN not in frame.columns:
        raise ValueError(f"{path.name} is missing {USER_COLUMN}")


def _forget_db(
    output: IOSettings,
    user_id: str,
    *,
    writer_lock: LockBackend | None,
) -> ForgetResult:
    table, _columns, user_col = recommendations_sql_names(
        output.options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
    )
    track_store = TrackStore(output, writer_lock=writer_lock)
    exposures = ExperimentStore(output, writer_lock=writer_lock)
    engine = _output_engine(require_option(output.options, "database_url", "db"))
    try:
        with engine.begin() as conn:
            recommendations = delete_rows_for_user(
                conn,
                table,
                user_id,
                fence=track_store._ensure_fence,
                user_column=user_col,
            )
            track = track_store.delete_user_rows(user_id, conn=conn)
            exposure_rows = exposures.delete_exposures_for_user(user_id, conn=conn)
            history = track_store.delete_history_for_user(user_id, conn=conn)
    finally:
        engine.dispose()
    return ForgetResult(
        recommendations=recommendations,
        track=track,
        exposures=exposure_rows,
        history=history,
    )


def _output_engine(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name != "sqlite":
        return engine

    # pysqlite commits a RELEASE SAVEPOINT unless BEGIN was sent explicitly.
    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection, _record) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin(conn) -> None:
        conn.exec_driver_sql("BEGIN")

    return engine


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
    try:
        frame = read_parquet(output.options, "recommendations.parquet")
    except FileNotFoundError:
        return 0
    if frame.empty or USER_COLUMN not in frame.columns:
        return 0
    return int((frame[USER_COLUMN].astype(str) == user_id).sum())
