"""Erase one user's recommendations, track rows, exposures, and history."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Connection, Engine

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.experiment.store import EXPOSURES_FILENAME, ExperimentStore, require_appendable_exposure_log
from cicerone.io.dataset_store import DatasetOutputSink
from cicerone.io.db_store import DEFAULT_RECOMMENDATIONS_TABLE, create_db_engine
from cicerone.io.jsonl_user import filter_user_lines
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

logger = logging.getLogger(__name__)

FORGET_OUTPUT_ERROR = 'forget-user requires output kind = "db" or a local dataset path'


@dataclass(frozen=True)
class ForgetResult:
    """Rows removed for one user. Missing files and tables count as zero."""

    recommendations: int
    track: int
    exposures: int
    history: int


@dataclass(frozen=True)
class _Swap:
    """Replace ``dest`` with ``staged``, or remove it when ``staged`` is None."""

    dest: Path
    staged: Path | None


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
    redis_url: str | None = None,
) -> ForgetResult:
    """Drop one user's output rows. Catalog users and events are left in place.

    Stored ids are compared after trimming, like event and track ingest.
    """
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("user_id is required")
    require_forget_output(output)
    if output.kind == "db":
        result = _forget_db(output, user_id, writer_lock=writer_lock)
    else:
        result = _forget_local(output, user_id, writer_lock=writer_lock)
    if not redis_url:
        return result
    # After the output erase: a Redis delete cannot join that transaction or swap.
    redis_store = TrackStore(output, writer_lock=writer_lock, redis_url=redis_url)
    redis_removed = redis_store.delete_redis_rows(user_id)
    return replace(result, track=result.track + redis_removed)


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
    engine = create_db_engine(require_option(output.options, "database_url", "db"), options=output.options)
    with _transaction(engine) as conn:
        recommendations = delete_rows_for_user(
            conn,
            table,
            user_id,
            fence=track_store.fence.ensure,
            user_column=user_col,
        )
        track = track_store.delete_user_rows(user_id, conn=conn)
        exposure_rows = exposures.delete_exposures_for_user(user_id, conn=conn)
        history = track_store.delete_history_for_user(user_id, conn=conn)
    return ForgetResult(
        recommendations=recommendations,
        track=track,
        exposures=exposure_rows,
        history=history,
    )


@contextmanager
def _transaction(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as conn:
        if engine.dialect.name != "sqlite":
            with conn.begin():
                yield conn
            return
        # pysqlite commits a RELEASE SAVEPOINT unless BEGIN was sent explicitly.
        # Set per connection: the engine is shared with the output writers.
        dbapi = conn.connection.dbapi_connection
        assert dbapi is not None
        previous = dbapi.isolation_level
        dbapi.isolation_level = None
        try:
            with conn.begin():
                conn.exec_driver_sql("BEGIN")
                yield conn
        finally:
            dbapi.isolation_level = previous


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
        # Writer lease first, then file locks: the same order as a job run.
        with (
            sink.recommendations_write(),
            exclusive_file_lock(root / ".track.jsonl.lock"),
            exclusive_file_lock(root / ".exposures.jsonl.lock"),
        ):
            ensure_writer_owned(writer_lock)
            swaps: list[_Swap] = []
            try:
                counts = _plan_local_erase(root, user_id, swaps)
                ensure_writer_owned(writer_lock)
                _commit_swaps(swaps)
            finally:
                _discard_staged(swaps)
            return counts

    if writer_lock_held_here(writer_lock):
        return _run()
    with held_writer_lock(writer_lock):
        return _run()


def _plan_local_erase(root: Path, user_id: str, swaps: list[_Swap]) -> ForgetResult:
    recommendations = _plan_parquet(root / "recommendations.parquet", user_id, swaps, drop_empty=False)
    track = _plan_jsonl(root / TRACK_FILENAME, user_id, swaps)
    exposures = _plan_jsonl(root / EXPOSURES_FILENAME, user_id, swaps)
    history_paths = [root / HISTORY_FILENAME]
    history_dir = root / HISTORY_DIR
    if history_dir.is_dir():
        history_paths.extend(sorted(history_dir.glob("*.parquet")))
    history = sum(_plan_parquet(path, user_id, swaps, drop_empty=True) for path in history_paths)
    return ForgetResult(recommendations, track, exposures, history)


def _plan_parquet(path: Path, user_id: str, swaps: list[_Swap], *, drop_empty: bool) -> int:
    if not path.is_file():
        return 0
    frame = pd.read_parquet(path)
    if frame.empty:
        return 0
    if USER_COLUMN not in frame.columns:
        raise ValueError(f"{path.name} is missing {USER_COLUMN}")
    mask = frame[USER_COLUMN].astype(str).str.strip() == user_id
    removed = int(mask.sum())
    if not removed:
        return 0
    kept = frame.loc[~mask].reset_index(drop=True)
    if kept.empty and drop_empty:
        swaps.append(_Swap(path, None))
        return removed
    staged = _new_temp_path(path, ".tmp")
    swaps.append(_Swap(path, staged))
    kept.to_parquet(staged, index=False)
    return removed


def _plan_jsonl(path: Path, user_id: str, swaps: list[_Swap]) -> int:
    if not path.is_file():
        return 0
    staged = _new_temp_path(path, ".tmp")
    swaps.append(_Swap(path, staged))
    with path.open("rb") as source, staged.open("wb") as sink:
        removed = filter_user_lines(source, sink, user_id)
    if not removed:
        swaps.pop()
        staged.unlink(missing_ok=True)
    return removed


def _new_temp_path(dest: Path, suffix: str) -> Path:
    """A fresh hidden file next to ``dest``; mkstemp creates it exclusively, so no symlink is followed.

    It gets ``dest``'s permission bits, since mkstemp's 0600 would hide a shared file once swapped in.
    """
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=suffix, dir=dest.parent)
    try:
        os.fchmod(fd, stat.S_IMODE(dest.stat().st_mode) & 0o777)
    finally:
        os.close(fd)
    return Path(name)


def _commit_swaps(swaps: list[_Swap]) -> None:
    """Swap every file in, keeping the originals until all swaps succeed.

    A crash mid-commit leaves the originals as ``.<name>.*.bak`` next to them.
    """
    moved: list[tuple[Path, Path]] = []
    try:
        for swap in swaps:
            backup = _new_temp_path(swap.dest, ".bak")
            try:
                os.replace(swap.dest, backup)
            except BaseException:
                backup.unlink(missing_ok=True)
                raise
            moved.append((backup, swap.dest))
            if swap.staged is not None:
                os.replace(swap.staged, swap.dest)
    except BaseException as exc:
        stranded = _restore(moved)
        if stranded:
            exc.add_note("originals not restored: " + ", ".join(str(path) for path in stranded))
        raise
    for backup, _dest in moved:
        try:
            backup.unlink()
        except OSError:
            logger.warning("Erase committed; could not remove backup %s", backup)


def _restore(moved: list[tuple[Path, Path]]) -> list[Path]:
    stranded: list[Path] = []
    for backup, dest in reversed(moved):
        try:
            os.replace(backup, dest)
        except OSError:
            stranded.append(backup)
    return stranded


def _discard_staged(swaps: list[_Swap]) -> None:
    for swap in swaps:
        if swap.staged is not None:
            swap.staged.unlink(missing_ok=True)
