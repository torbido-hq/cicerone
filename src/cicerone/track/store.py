"""Persist impression/click rows, eval reports, and optional rec history."""

from __future__ import annotations

import fcntl
import json
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, bindparam, create_engine, text

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.io.db_errors import is_missing_table_error
from cicerone.io.db_store import MISSING_TABLE_ERRORS
from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    require_option,
    sql_identifier,
    storage_backend,
    validate_storage_options,
)
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)

logger = logging.getLogger(__name__)

TRACK_FILENAME = "track.jsonl"
EVAL_FILENAME = "track_eval.json"
HISTORY_FILENAME = "recommendation_history.parquet"
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


class TrackStore:
    """Output-store side channel for track rows, eval JSON, and rec snapshots."""

    def __init__(self, output: IOSettings):
        self._output = output
        self._kind = output.kind
        self._options = output.options
        self._engine: Engine | None = None
        self._known_ids: set[str] | None = None
        self._track_size: int | None = None
        self._append_lock = threading.Lock()

    def _db_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                require_option(self._options, "database_url", "db"), pool_pre_ping=True
            )
        return self._engine

    def append_rows(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        payload = [dict(row) for row in rows]
        if self._kind == "db":
            return self._append_rows_db(payload)
        require_appendable_track_log(self._output)
        with self._dataset_append_lock():
            known = self._refresh_known_ids()
            fresh: list[dict[str, Any]] = []
            for row in payload:
                event_id = str(row.get("event_id") or "")
                if event_id and event_id in known:
                    continue
                if event_id:
                    known.add(event_id)
                fresh.append(row)
            if not fresh:
                return 0
            encoded = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fresh).encode("utf-8")
            self._append_bytes(TRACK_FILENAME, encoded)
            self._track_size = (self._track_size or 0) + len(encoded)
            return len(fresh)

    def read_rows(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self._read_rows_db() if self._kind == "db" else self._read_rows_dataset()
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            if kind is not None and str(row.get("kind") or "") != kind:
                continue
            unique.append(row)
        return unique

    def write_eval(self, report: Mapping[str, Any]) -> None:
        payload = dict(report)
        if self._kind == "db":
            self._write_eval_db(payload)
            return
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self._write_bytes(EVAL_FILENAME, encoded, "application/json")

    def read_eval(self) -> dict[str, Any] | None:
        if self._kind == "db":
            return self._read_eval_db()
        raw = self._read_bytes(EVAL_FILENAME)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid track_eval.json; ignoring")
            return None
        return parsed if isinstance(parsed, dict) else None

    def append_history(self, recommendations: pd.DataFrame, *, generated_at: str) -> None:
        if recommendations.empty:
            return
        frame = _history_frame(recommendations, generated_at)
        if self._kind == "db":
            self._append_history_db(frame)
            return
        existing = self.read_history()
        merged = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame
        buf = BytesIO()
        merged.to_parquet(buf, index=False)
        self._write_bytes(HISTORY_FILENAME, buf.getvalue(), "application/octet-stream")

    def read_history(self) -> pd.DataFrame:
        if self._kind == "db":
            return self._read_history_db()
        from cicerone.io.options import read_parquet

        try:
            return read_parquet(self._options, HISTORY_FILENAME)
        except FileNotFoundError:
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        except Exception as exc:
            if is_s3_not_found(exc):
                return pd.DataFrame(columns=list(HISTORY_COLUMNS))
            raise

    def _ensure_track_table(self, conn: Any, table: str) -> None:
        conn.execute(
            text(
                f'CREATE TABLE IF NOT EXISTS "{table}" ('
                "event_id TEXT PRIMARY KEY, "
                "kind TEXT NOT NULL, "
                "user_id TEXT NOT NULL, "
                "item_id TEXT NOT NULL, "
                "rank INTEGER, "
                "occurred_at TEXT NOT NULL, "
                "variant TEXT, "
                "experiment_id TEXT, "
                "generated_at TEXT"
                ")"
            )
        )

    def _append_rows_db(self, rows: list[dict[str, Any]]) -> int:
        table = sql_identifier(
            self._options.get("track_table", DEFAULT_TRACK_TABLE),
            option="track_table",
        )
        engine = self._db_engine()
        insert_sql = text(
            f'INSERT INTO "{table}" (event_id, kind, user_id, item_id, rank, occurred_at, '
            "variant, experiment_id, generated_at) "
            "VALUES (:event_id, :kind, :user_id, :item_id, :rank, :occurred_at, "
            ":variant, :experiment_id, :generated_at) "
            "ON CONFLICT (event_id) DO NOTHING"
        )
        params: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            params.append(
                {
                    "event_id": event_id,
                    "kind": str(row.get("kind") or ""),
                    "user_id": str(row.get("user_id") or ""),
                    "item_id": str(row.get("item_id") or ""),
                    "rank": row.get("rank"),
                    "occurred_at": str(row.get("occurred_at") or ""),
                    "variant": row.get("variant"),
                    "experiment_id": row.get("experiment_id"),
                    "generated_at": row.get("generated_at"),
                }
            )
        with engine.begin() as conn:
            self._ensure_track_table(conn, table)
            existing = _existing_event_ids(conn, table, [row["event_id"] for row in params])
            fresh = [row for row in params if not row["event_id"] or row["event_id"] not in existing]
            if not fresh:
                return 0
            result = conn.execute(insert_sql, fresh)
            rowcount = result.rowcount
            if rowcount is not None and rowcount >= 0:
                return int(rowcount)
            return len(fresh)

    def _read_rows_db(self) -> list[dict[str, Any]]:
        table = sql_identifier(
            self._options.get("track_table", DEFAULT_TRACK_TABLE),
            option="track_table",
        )
        engine = self._db_engine()
        try:
            frame = pd.read_sql(text(f'SELECT * FROM "{table}"'), engine)
        except MISSING_TABLE_ERRORS:
            return []
        except Exception as exc:
            if is_missing_table_error(exc):
                return []
            logger.exception("Failed to read track table %r", table)
            return []
        if frame.empty:
            return []
        records = frame.to_dict(orient="records")
        return [{str(key): _jsonish(value) for key, value in row.items()} for row in records]

    def _read_rows_dataset(self) -> list[dict[str, Any]]:
        raw = self._read_bytes(TRACK_FILENAME)
        if raw is None:
            if self._known_ids is None:
                self._known_ids = set()
            return []
        rows: list[dict[str, Any]] = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        if self._known_ids is None:
            self._known_ids = {str(row.get("event_id") or "") for row in rows}
            self._known_ids.discard("")
        return rows

    def _track_file_size(self) -> int:
        path = Path(require_option(self._options, "path", "local")) / TRACK_FILENAME
        if not path.exists():
            return 0
        return path.stat().st_size

    def _refresh_known_ids(self) -> set[str]:
        size = self._track_file_size()
        if self._known_ids is not None and self._track_size == size:
            return self._known_ids
        self._known_ids = None
        self._read_rows_dataset()
        assert self._known_ids is not None
        self._track_size = size
        return self._known_ids

    @contextmanager
    def _dataset_append_lock(self) -> Iterator[None]:
        path = Path(require_option(self._options, "path", "local")) / ".track.jsonl.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._append_lock, path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_eval_db(self, payload: dict[str, Any]) -> None:
        table = sql_identifier(
            self._options.get("eval_table", DEFAULT_EVAL_TABLE),
            option="eval_table",
        )
        engine = self._db_engine()
        encoded = json.dumps(payload)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f'CREATE TABLE IF NOT EXISTS "{table}" (payload TEXT NOT NULL, written_at TEXT NOT NULL)'
                )
            )
            conn.execute(text(f'DELETE FROM "{table}"'))
            conn.execute(
                text(f'INSERT INTO "{table}" (payload, written_at) VALUES (:payload, :written_at)'),
                {"payload": encoded, "written_at": pd.Timestamp.now(tz="UTC").isoformat()},
            )

    def _read_eval_db(self) -> dict[str, Any] | None:
        table = sql_identifier(
            self._options.get("eval_table", DEFAULT_EVAL_TABLE),
            option="eval_table",
        )
        engine = self._db_engine()
        try:
            frame = pd.read_sql(text(f'SELECT payload FROM "{table}" LIMIT 1'), engine)
        except MISSING_TABLE_ERRORS:
            return None
        except Exception as exc:
            if is_missing_table_error(exc):
                return None
            logger.exception("Failed to read eval table %r", table)
            return None
        if frame.empty:
            return None
        raw = frame.iloc[0]["payload"]
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _append_history_db(self, frame: pd.DataFrame) -> None:
        table = sql_identifier(
            self._options.get("history_table", DEFAULT_HISTORY_TABLE),
            option="history_table",
        )
        frame.to_sql(table, self._db_engine(), if_exists="append", index=False)

    def _read_history_db(self) -> pd.DataFrame:
        table = sql_identifier(
            self._options.get("history_table", DEFAULT_HISTORY_TABLE),
            option="history_table",
        )
        engine = self._db_engine()
        try:
            return pd.read_sql(text(f'SELECT * FROM "{table}"'), engine)
        except MISSING_TABLE_ERRORS:
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        except Exception as exc:
            if is_missing_table_error(exc):
                return pd.DataFrame(columns=list(HISTORY_COLUMNS))
            logger.exception("Failed to read history table %r", table)
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))

    def _read_bytes(self, filename: str) -> bytes | None:
        backend = validate_storage_options(self._options)
        if backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            if not path.exists():
                return None
            return path.read_bytes()
        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, filename)
        client = build_s3_client(self._options)
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            raise
        return obj["Body"].read()

    def _write_bytes(self, filename: str, payload: bytes, content_type: str) -> None:
        backend = validate_storage_options(self._options)
        if backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
            return
        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, filename)
        client = build_s3_client(self._options)
        client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)

    def _append_bytes(self, filename: str, payload: bytes) -> None:
        path = Path(require_option(self._options, "path", "local")) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(payload)


def _existing_event_ids(conn: Any, table: str, event_ids: Sequence[str]) -> set[str]:
    ids = [event_id for event_id in event_ids if event_id]
    if not ids:
        return set()
    stmt = text(f'SELECT event_id FROM "{table}" WHERE event_id IN :ids').bindparams(
        bindparam("ids", expanding=True)
    )
    return {str(row[0]) for row in conn.execute(stmt, {"ids": ids}) if row[0] is not None}


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
