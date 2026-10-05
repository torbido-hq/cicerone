"""Erase one user's recommendations, track rows, exposures, and history."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.experiment.store import EXPOSURES_FILENAME, ExperimentStore, require_appendable_exposure_log
from cicerone.io.dataset_store import DatasetOutputSink
from cicerone.io.db_store import DEFAULT_RECOMMENDATIONS_TABLE
from cicerone.io.jsonl_user import drop_user_lines
from cicerone.io.options import exclusive_file_lock, require_option, storage_backend
from cicerone.io.recommendation_schema import USER_COLUMN, recommendations_sql_names
from cicerone.locks.hold import (
    LockBackend,
    ensure_writer_owned,
    held_writer_lock,
    writer_lock_held_here,
)
from cicerone.track.store import TrackStore, require_appendable_track_log
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
    return _forget_local(output, user_id, writer_lock=writer_lock)


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


def _forget_local(
    output: IOSettings,
    user_id: str,
    *,
    writer_lock: LockBackend | None,
) -> ForgetResult:
    require_appendable_track_log(output)
    require_appendable_exposure_log(output)
    root = Path(require_option(output.options, "path", "local"))
    sink = DatasetOutputSink(output.options, writer_lock=writer_lock)

    def _run() -> ForgetResult:
        with (
            sink.recommendations_write(),
            exclusive_file_lock(root / ".track.jsonl.lock"),
            exclusive_file_lock(root / ".exposures.jsonl.lock"),
        ):
            ensure_writer_owned(writer_lock)
            rewrites, unlinks, counts = _plan_local_erase(root, user_id)
            ensure_writer_owned(writer_lock)
            _commit_local_rewrites(rewrites, unlinks)
            return counts

    if writer_lock_held_here(writer_lock):
        return _run()
    with held_writer_lock(writer_lock):
        return _run()


def _plan_local_erase(root: Path, user_id: str) -> tuple[list[tuple[Path, bytes]], list[Path], ForgetResult]:
    rewrites: list[tuple[Path, bytes]] = []
    unlinks: list[Path] = []
    recommendations = _plan_recommendations(root, user_id, rewrites)
    track = _plan_jsonl(root / TRACK_FILENAME, user_id, rewrites)
    exposures = _plan_jsonl(root / EXPOSURES_FILENAME, user_id, rewrites)
    history = _plan_history(root, user_id, rewrites, unlinks)
    return rewrites, unlinks, ForgetResult(recommendations, track, exposures, history)


def _plan_recommendations(root: Path, user_id: str, rewrites: list[tuple[Path, bytes]]) -> int:
    path = root / "recommendations.parquet"
    if not path.is_file():
        return 0
    frame = pd.read_parquet(path)
    _require_user_column(path, frame)
    if frame.empty:
        return 0
    mask = frame[USER_COLUMN].astype(str) == user_id
    removed = int(mask.sum())
    if removed:
        kept = frame.loc[~mask].reset_index(drop=True)
        rewrites.append((path, _parquet_bytes(kept)))
    return removed


def _plan_jsonl(path: Path, user_id: str, rewrites: list[tuple[Path, bytes]]) -> int:
    if not path.is_file():
        return 0
    payload, removed = drop_user_lines(path.read_bytes(), user_id)
    if removed:
        rewrites.append((path, payload))
    return removed


def _plan_history(
    root: Path,
    user_id: str,
    rewrites: list[tuple[Path, bytes]],
    unlinks: list[Path],
) -> int:
    paths = [root / HISTORY_FILENAME]
    history = root / HISTORY_DIR
    if history.is_dir():
        paths.extend(sorted(history.glob("*.parquet")))
    total = 0
    for path in paths:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        _require_user_column(path, frame)
        if frame.empty:
            continue
        mask = frame[USER_COLUMN].astype(str) == user_id
        removed = int(mask.sum())
        if not removed:
            continue
        total += removed
        kept = frame.loc[~mask].reset_index(drop=True)
        if kept.empty:
            unlinks.append(path)
        else:
            rewrites.append((path, _parquet_bytes(kept)))
    return total


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _commit_local_rewrites(rewrites: list[tuple[Path, bytes]], unlinks: list[Path]) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for dest, payload in rewrites:
            tmp = dest.with_name(f".{dest.name}.tmp")
            tmp.write_bytes(payload)
            staged.append((tmp, dest))
    except OSError:
        _unlink_temps(staged)
        raise
    replaced = 0
    try:
        for tmp, dest in staged:
            tmp.replace(dest)
            replaced += 1
        for path in unlinks:
            path.unlink()
    except OSError:
        _unlink_temps(staged[replaced:])
        raise


def _unlink_temps(staged: list[tuple[Path, Path]]) -> None:
    for tmp, _dest in staged:
        tmp.unlink(missing_ok=True)
