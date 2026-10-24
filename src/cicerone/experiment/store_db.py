"""Database backend for ExperimentStore."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError

from cicerone.io.db_errors import SQL_READ_ERRORS, is_missing_column_error, is_missing_table_error
from cicerone.io.engines import create_table_sql, engine_for, insert_sql, quote_identifier
from cicerone.io.options import require_option, sql_identifier, table_option
from cicerone.locks import WriteFence

logger = logging.getLogger(__name__)

DEFAULT_EXPOSURES_TABLE = "recommendation_exposures"
DEFAULT_STATE_TABLE = "experiment_state"
STATE_CORE_KEYS: tuple[str, ...] = ("experiment_id", "promoted_variant", "promoted_at")
EXPOSURE_COLUMNS: tuple[str, ...] = (
    "user_id",
    "experiment_id",
    "variant",
    "generated_at",
    "exposed_at",
)


_STATE_TABLE = (
    ("experiment_id", "TEXT PRIMARY KEY"),
    ("promoted_variant", "TEXT"),
    ("promoted_at", "TEXT"),
    ("payload", "TEXT"),
)


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


def jsonish(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value


class ExperimentDbBackend:
    """Promote state in one table (one row), exposures in another."""

    def __init__(self, options: dict[str, Any], fence: WriteFence):
        self._options = options
        self._fence = fence
        self._engine: Engine | None = None

    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = engine_for(require_option(self._options, "database_url", "db"))
        return self._engine

    def read_state(self) -> dict[str, Any] | None:
        table = table_option(self._options, "experiment_state_table", DEFAULT_STATE_TABLE)
        engine = self.engine()
        quoted_table, promoted_at = quote_identifier(engine, table), quote_identifier(engine, "promoted_at")
        try:
            frame = pd.read_sql(
                text(
                    f"SELECT * FROM {quoted_table} "
                    f"ORDER BY ({promoted_at} IS NULL), {promoted_at} DESC LIMIT 1"
                ),
                engine,
            )
        except SQL_READ_ERRORS as exc:
            if is_missing_column_error(exc):
                try:
                    frame = pd.read_sql(text(f"SELECT * FROM {quoted_table} LIMIT 1"), engine)
                except SQL_READ_ERRORS as retry_exc:
                    if is_missing_table_error(retry_exc) or is_missing_column_error(retry_exc):
                        return None
                    logger.exception("Failed to read experiment state table %r", table)
                    raise
            elif is_missing_table_error(exc):
                return None
            else:
                logger.exception("Failed to read experiment state table %r", table)
                raise
        if frame.empty:
            return None
        return _hydrate_state_row(
            {str(key): jsonish(value) for key, value in frame.iloc[0].to_dict().items()}
        )

    def write_state(self, state: dict[str, Any]) -> None:
        self.state_writer(state)()

    def state_writer(self, state: dict[str, Any]) -> Callable[[], None]:
        # Encode before the caller takes the fence; only the SQL runs under it.
        table = table_option(self._options, "experiment_state_table", DEFAULT_STATE_TABLE)
        engine = self.engine()
        quoted_table = quote_identifier(engine, table)
        experiment_id = str(state.get("experiment_id") or "")
        promoted_variant = state.get("promoted_variant")
        promoted_at = state.get("promoted_at")
        if promoted_variant is not None:
            promoted_variant = str(promoted_variant)
        if promoted_at is not None:
            promoted_at = str(promoted_at)
        create_sql = text(create_table_sql(engine, table, _STATE_TABLE))
        params = {
            "experiment_id": experiment_id,
            "promoted_variant": promoted_variant,
            "promoted_at": promoted_at,
            "payload": _state_payload_json(state),
        }
        row_sql = text(insert_sql(engine, table, list(params)))

        def _fence() -> None:
            self._fence.ensure()

        def _replace_row(conn: Any) -> None:
            _fence()
            conn.execute(text(f"DELETE FROM {quoted_table}"))
            _fence()
            conn.execute(row_sql, params)
            _fence()

        def _write() -> None:
            try:
                with engine.begin() as conn:
                    _fence()
                    conn.execute(create_sql)
                    _replace_row(conn)
            except SQLAlchemyError as exc:
                if not is_missing_column_error(exc):
                    raise
                with engine.begin() as conn:
                    _fence()
                    payload = quote_identifier(engine, "payload")
                    conn.execute(text(f"ALTER TABLE {quoted_table} ADD COLUMN {payload} TEXT"))
                    _replace_row(conn)

        return _write

    def exposure_writer(self, rows: Sequence[Mapping[str, Any]]) -> Callable[[], None]:
        table = table_option(self._options, "exposures_table", DEFAULT_EXPOSURES_TABLE)
        engine = self.engine()
        frame = pd.DataFrame(list(rows))

        def _append() -> None:
            with engine.begin() as conn:
                self._fence.ensure()
                frame.to_sql(table, conn, if_exists="append", index=False)
                self._fence.ensure()

        return _append

    def append_exposures(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.exposure_writer(rows)()

    def stored_exposure_columns(self, engine: Engine, table: str) -> list[str] | None:
        try:
            existing = {col["name"] for col in inspect(engine).get_columns(table)}
        except NoSuchTableError:
            return None
        except SQL_READ_ERRORS as exc:
            if is_missing_table_error(exc):
                return None
            logger.exception("Failed to inspect exposures table %r", table)
            raise
        return [column for column in EXPOSURE_COLUMNS if column in existing]

    def read_exposures(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        table = table_option(self._options, "exposures_table", DEFAULT_EXPOSURES_TABLE)
        engine = self.engine()
        quoted_table = quote_identifier(engine, table)
        selected = self.stored_exposure_columns(engine, table)
        if selected is not None and not selected:
            logger.warning("Exposures table %r has none of the exposure columns; ignoring rows", table)
            return []
        if experiment_id and selected is not None and "experiment_id" not in selected:
            logger.warning("Exposures table %r has no experiment_id column; ignoring rows", table)
            return []
        # A missing table falls through to the read, whose error handling decides.
        quoted = (
            "*"
            if selected is None
            else ", ".join(
                quote_identifier(engine, sql_identifier(column, option="exposures column"))
                for column in selected
            )
        )
        if experiment_id:
            sql = text(
                f"SELECT {quoted} FROM {quoted_table} "
                f"WHERE {quote_identifier(engine, 'experiment_id')} = :experiment_id"
            )
            params: dict[str, Any] = {"experiment_id": experiment_id}
        else:
            sql = text(f"SELECT {quoted} FROM {quoted_table}")
            params = {}
        try:
            frame = pd.read_sql(sql, engine, params=params)
        except SQL_READ_ERRORS as exc:
            if is_missing_table_error(exc):
                return []
            # The table may appear after inspection without an experiment_id column.
            if experiment_id and selected is None and is_missing_column_error(exc):
                logger.warning("Exposures table %r has no experiment_id column; ignoring rows", table)
                return []
            logger.exception("Failed to read exposures table %r", table)
            raise
        if frame.empty:
            return []
        records = frame.to_dict(orient="records")
        return [{str(key): jsonish(value) for key, value in row.items()} for row in records]
