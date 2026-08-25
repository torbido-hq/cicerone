"""Persist experiment promote state and optional serve-time exposures."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.config import IOSettings
from cicerone.io.db_errors import is_missing_table_error
from cicerone.io.db_store import MISSING_TABLE_ERRORS
from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    require_option,
    sql_identifier,
    validate_storage_options,
)

logger = logging.getLogger(__name__)

STATE_FILENAME = "experiment_state.json"
EXPOSURES_FILENAME = "exposures.jsonl"
DEFAULT_EXPOSURES_TABLE = "recommendation_exposures"
DEFAULT_STATE_TABLE = "experiment_state"

EXPOSURE_COLUMNS: tuple[str, ...] = (
    "user_id",
    "experiment_id",
    "variant",
    "generated_at",
    "exposed_at",
)


def experiment_state(
    experiment_id: str,
    *,
    promoted_variant: str | None,
    promoted_at: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "promoted_variant": promoted_variant,
        "promoted_at": promoted_at or (datetime.now(UTC).isoformat() if promoted_variant else None),
    }


class ExperimentStore:
    """Output-store side channel for promote state and exposure logs."""

    def __init__(self, output: IOSettings):
        self._output = output
        self._kind = output.kind
        self._options = output.options

    def read_state(self) -> dict[str, Any] | None:
        if self._kind == "db":
            return self._read_state_db()
        return self._read_state_dataset()

    def write_state(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        if self._kind == "db":
            self._write_state_db(payload)
            return
        self._write_bytes(STATE_FILENAME, json.dumps(payload, indent=2).encode("utf-8"), "application/json")

    def append_exposures(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        if self._kind == "db":
            self._append_exposures_db(rows)
            return
        payload = "".join(json.dumps(dict(row), separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
        self._append_bytes(EXPOSURES_FILENAME, payload)

    def read_exposures(self) -> list[dict[str, Any]]:
        if self._kind == "db":
            return self._read_exposures_db()
        return self._read_exposures_dataset()

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
        engine = create_engine(require_option(self._options, "database_url", "db"), pool_pre_ping=True)
        try:
            frame = pd.read_sql(text(f'SELECT * FROM "{table}" ORDER BY rowid DESC LIMIT 1'), engine)
        except MISSING_TABLE_ERRORS:
            return None
        except Exception as exc:
            if is_missing_table_error(exc):
                return None
            # Postgres has no rowid — fall back to unordered last insert via ctid-less LIMIT.
            try:
                frame = pd.read_sql(text(f'SELECT * FROM "{table}" LIMIT 1'), engine)
            except Exception:
                logger.exception("Failed to read experiment state table %r", table)
                return None
        if frame.empty:
            return None
        return {str(key): _jsonish(value) for key, value in frame.iloc[0].to_dict().items()}

    def _write_state_db(self, state: dict[str, Any]) -> None:
        table = sql_identifier(
            self._options.get("experiment_state_table", DEFAULT_STATE_TABLE),
            option="experiment_state_table",
        )
        engine = create_engine(require_option(self._options, "database_url", "db"), pool_pre_ping=True)
        frame = pd.DataFrame([state])
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
            frame.to_sql(table, conn, if_exists="append", index=False)

    def _append_exposures_db(self, rows: Sequence[Mapping[str, Any]]) -> None:
        table = sql_identifier(
            self._options.get("exposures_table", DEFAULT_EXPOSURES_TABLE),
            option="exposures_table",
        )
        engine = create_engine(require_option(self._options, "database_url", "db"), pool_pre_ping=True)
        pd.DataFrame(list(rows)).to_sql(table, engine, if_exists="append", index=False)

    def _read_exposures_db(self) -> list[dict[str, Any]]:
        table = sql_identifier(
            self._options.get("exposures_table", DEFAULT_EXPOSURES_TABLE),
            option="exposures_table",
        )
        engine = create_engine(require_option(self._options, "database_url", "db"), pool_pre_ping=True)
        try:
            frame = pd.read_sql(text(f'SELECT * FROM "{table}"'), engine)
        except MISSING_TABLE_ERRORS:
            return []
        except Exception as exc:
            if is_missing_table_error(exc):
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
        backend = validate_storage_options(self._options)
        if backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(payload)
            return
        existing = self._read_bytes(filename) or b""
        self._write_bytes(filename, existing + payload, "application/x-ndjson")


def _jsonish(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value
