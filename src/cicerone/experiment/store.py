"""Persist experiment promote state and optional serve-time exposures."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.io.db_errors import is_missing_column_error, is_missing_table_error
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

logger = logging.getLogger(__name__)

STATE_FILENAME = "experiment_state.json"
EXPOSURES_FILENAME = "exposures.jsonl"
DEFAULT_EXPOSURES_TABLE = "recommendation_exposures"
DEFAULT_STATE_TABLE = "experiment_state"
EXPOSURE_LOG_BACKEND_ERROR = (
    'experiment.log_exposures requires output kind = "db" or a local dataset path; '
    "object-store JSONL append is not atomic"
)
EXPOSURE_LOG_HA_ERROR = 'experiment.log_exposures with events.ha requires output kind = "db"'

EXPOSURE_COLUMNS: tuple[str, ...] = (
    "user_id",
    "experiment_id",
    "variant",
    "generated_at",
    "exposed_at",
)


def require_appendable_exposure_log(output: IOSettings) -> None:
    """Reject object-store JSONL append (read-modify-write is racy)."""
    if output.kind == "db":
        return
    if output.kind == "dataset" and storage_backend(output.options) == "local":
        return
    raise ConfigError(EXPOSURE_LOG_BACKEND_ERROR)


STATE_CORE_KEYS: tuple[str, ...] = ("experiment_id", "promoted_variant", "promoted_at")


def experiment_state(
    experiment_id: str,
    *,
    promoted_variant: str | None,
    promoted_at: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "promoted_variant": promoted_variant,
        "promoted_at": promoted_at
        if promoted_at is not None
        else (datetime.now(UTC).isoformat() if promoted_variant else None),
    }
    for key, value in extra.items():
        if key not in STATE_CORE_KEYS and key != "payload":
            payload[key] = value
    return payload


def merge_experiment_state(
    previous: Mapping[str, Any] | None,
    *,
    experiment_id: str,
    promoted_variant: str | None,
    promoted_at: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    merged = dict(previous or {})
    merged.update(
        experiment_state(
            experiment_id,
            promoted_variant=promoted_variant,
            promoted_at=promoted_at,
            **extra,
        )
    )
    return merged


def active_pair_from_state(state: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if not state:
        return None
    champion = str(state.get("champion") or "")
    challenger = str(state.get("challenger") or "")
    if champion and challenger:
        return champion, challenger
    return None


def _state_payload_json(state: Mapping[str, Any]) -> str:
    extra = {key: value for key, value in state.items() if key not in STATE_CORE_KEYS and key != "payload"}
    return json.dumps(extra, separators=(",", ":"))


def _hydrate_state_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.pop("payload", None)
    parsed: dict[str, Any] | None = None
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded
    if parsed:
        for key, value in parsed.items():
            if key not in STATE_CORE_KEYS and key != "payload":
                row[key] = value
    return row


class ExperimentStore:
    """Output-store side channel for promote state and exposure logs."""

    def __init__(self, output: IOSettings):
        self._output = output
        self._kind = output.kind
        self._options = output.options
        self._engine: Engine | None = None
        self._promote_lock = threading.Lock()
        self._promote_loaded = False
        self._promote_experiment_id: str | None = None
        self._promote_value: str | None = None
        self._promote_state: dict[str, Any] | None = None

    def _db_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                require_option(self._options, "database_url", "db"), pool_pre_ping=True
            )
        return self._engine

    def read_state(self) -> dict[str, Any] | None:
        state = self._read_state_db() if self._kind == "db" else self._read_state_dataset()
        self._remember_state(state)
        return state

    def last_state(self, experiment_id: str) -> dict[str, Any] | None:
        """Last successful promote-state row for ``experiment_id``, if any."""
        wanted = str(experiment_id)
        with self._promote_lock:
            if not self._promote_loaded or self._promote_state is None:
                return None
            if str(self._promote_state.get("experiment_id") or "") != wanted:
                return None
            return dict(self._promote_state)

    def _remember_state(self, state: dict[str, Any] | None) -> None:
        with self._promote_lock:
            self._promote_loaded = True
            if state and state.get("experiment_id"):
                self._promote_experiment_id = str(state.get("experiment_id") or "")
                promoted = state.get("promoted_variant")
                self._promote_value = str(promoted) if promoted else None
                self._promote_state = dict(state)
            else:
                self._promote_experiment_id = None
                self._promote_value = None
                self._promote_state = None

    def promoted_variant(self, experiment_id: str) -> str | None:
        """Return the live promote winner; reuse the last successful read on failure."""
        return self.assignment_overlay(experiment_id)[0]

    def assignment_overlay(self, experiment_id: str) -> tuple[str | None, tuple[str, str] | None]:
        """Return ``(promoted_variant, active_pair)``; reuse the last successful read on failure."""
        wanted = str(experiment_id)
        try:
            state = self.read_state()
        except Exception:
            logger.exception("Failed to read experiment promote state")
            with self._promote_lock:
                if self._promote_loaded and self._promote_experiment_id == wanted:
                    return self._promote_value, active_pair_from_state(self._promote_state)
            return None, None
        if state and str(state.get("experiment_id") or "") == wanted:
            promoted = state.get("promoted_variant")
            return (str(promoted) if promoted else None), active_pair_from_state(state)
        return None, None

    def write_state(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        if self._kind == "db":
            self._write_state_db(payload)
        else:
            encoded = json.dumps(payload, indent=2).encode("utf-8")
            self._write_bytes(STATE_FILENAME, encoded, "application/json")
        promoted = payload.get("promoted_variant")
        with self._promote_lock:
            self._promote_loaded = True
            self._promote_experiment_id = str(payload.get("experiment_id") or "")
            self._promote_value = str(promoted) if promoted else None
            self._promote_state = dict(payload)

    def append_exposures(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        if self._kind == "db":
            self._append_exposures_db(rows)
            return
        require_appendable_exposure_log(self._output)
        payload = "".join(json.dumps(dict(row), separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
        self._append_bytes(EXPOSURES_FILENAME, payload)

    def read_exposures(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if self._kind == "db":
            rows = self._read_exposures_db(experiment_id=experiment_id)
        else:
            rows = self._read_exposures_dataset()
        if not experiment_id:
            return rows
        return [row for row in rows if str(row.get("experiment_id") or "") == experiment_id]

    def _read_state_dataset(self) -> dict[str, Any] | None:
        raw = self._read_bytes(STATE_FILENAME)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid experiment_state.json; ignoring")
            return None
        return parsed if isinstance(parsed, dict) else None

    def _read_state_db(self) -> dict[str, Any] | None:
        table = sql_identifier(
            self._options.get("experiment_state_table", DEFAULT_STATE_TABLE),
            option="experiment_state_table",
        )
        engine = self._db_engine()
        try:
            frame = pd.read_sql(
                text(f'SELECT * FROM "{table}" ORDER BY (promoted_at IS NULL), promoted_at DESC LIMIT 1'),
                engine,
            )
        except Exception as exc:
            if is_missing_column_error(exc):
                try:
                    frame = pd.read_sql(text(f'SELECT * FROM "{table}" LIMIT 1'), engine)
                except Exception:
                    logger.exception("Failed to read experiment state table %r", table)
                    raise
            elif isinstance(exc, MISSING_TABLE_ERRORS) or is_missing_table_error(exc):
                return None
            else:
                logger.exception("Failed to read experiment state table %r", table)
                raise
        if frame.empty:
            return None
        return _hydrate_state_row(
            {str(key): _jsonish(value) for key, value in frame.iloc[0].to_dict().items()}
        )

    def _write_state_db(self, state: dict[str, Any]) -> None:
        table = sql_identifier(
            self._options.get("experiment_state_table", DEFAULT_STATE_TABLE),
            option="experiment_state_table",
        )
        engine = self._db_engine()
        experiment_id = str(state.get("experiment_id") or "")
        promoted_variant = state.get("promoted_variant")
        promoted_at = state.get("promoted_at")
        if promoted_variant is not None:
            promoted_variant = str(promoted_variant)
        if promoted_at is not None:
            promoted_at = str(promoted_at)
        create_sql = text(
            f'CREATE TABLE IF NOT EXISTS "{table}" ('
            "experiment_id TEXT PRIMARY KEY, "
            "promoted_variant TEXT, "
            "promoted_at TEXT, "
            "payload TEXT"
            ")"
        )
        params = {
            "experiment_id": experiment_id,
            "promoted_variant": promoted_variant,
            "promoted_at": promoted_at,
            "payload": _state_payload_json(state),
        }
        insert_sql = text(
            f'INSERT INTO "{table}" (experiment_id, promoted_variant, promoted_at, payload) '
            "VALUES (:experiment_id, :promoted_variant, :promoted_at, :payload)"
        )
        with engine.begin() as conn:
            conn.execute(create_sql)
        try:
            with engine.begin() as conn:
                conn.execute(text(f'DELETE FROM "{table}"'))
                conn.execute(insert_sql, params)
        except Exception as exc:
            if not is_missing_column_error(exc):
                raise
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN payload TEXT'))
            with engine.begin() as conn:
                conn.execute(text(f'DELETE FROM "{table}"'))
                conn.execute(insert_sql, params)

    def _append_exposures_db(self, rows: Sequence[Mapping[str, Any]]) -> None:
        table = sql_identifier(
            self._options.get("exposures_table", DEFAULT_EXPOSURES_TABLE),
            option="exposures_table",
        )
        engine = self._db_engine()
        pd.DataFrame(list(rows)).to_sql(table, engine, if_exists="append", index=False)

    def _read_exposures_db(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        table = sql_identifier(
            self._options.get("exposures_table", DEFAULT_EXPOSURES_TABLE),
            option="exposures_table",
        )
        engine = self._db_engine()
        if experiment_id:
            sql = text(f'SELECT * FROM "{table}" WHERE experiment_id = :experiment_id')
            params: dict[str, Any] = {"experiment_id": experiment_id}
        else:
            sql = text(f'SELECT * FROM "{table}"')
            params = {}
        try:
            frame = pd.read_sql(sql, engine, params=params)
        except MISSING_TABLE_ERRORS:
            return []
        except Exception as exc:
            if is_missing_table_error(exc):
                return []
            if experiment_id and is_missing_column_error(exc):
                logger.warning("Exposures table %r has no experiment_id column; ignoring rows", table)
                return []
            logger.exception("Failed to read exposures table %r", table)
            return []
        if frame.empty:
            return []
        records = frame.to_dict(orient="records")
        return [{str(key): _jsonish(value) for key, value in row.items()} for row in records]

    def _read_exposures_dataset(self) -> list[dict[str, Any]]:
        raw = self._read_bytes(EXPOSURES_FILENAME)
        if raw is None:
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
        return rows

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


def _jsonish(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value
