"""Database input/output via SQLAlchemy.

Options: database_url (required); optional table names / raw SQL overrides
(events_query, users_query, items_query). Identifiers come from trusted
deploy-time config only.

NOTE: upgrading an existing recommendations table for optional ``reasons``
needs ``ALTER TABLE … ADD COLUMN reasons TEXT``; pandas to_sql(append) will
not add the column. Experiments similarly need ``ALTER TABLE … ADD COLUMN variant TEXT``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    LargeBinary,
    MetaData,
    Table,
    bindparam,
    create_engine,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import OperationalError, ProgrammingError

from cicerone.io.db_errors import is_missing_column_error
from cicerone.io.options import readonly_select, require_option, sql_identifier
from cicerone.io.recommendation_schema import (
    REASONS_COLUMN,
    VARIANT_COLUMN,
    recommendations_sql_names,
)
from cicerone.io.replace_users import RecommendationSchemaError, normalize_replace_user_ids
from cicerone.io.user_lookup import OCCURRED_AT_COLUMN, filter_rows_for_user, newest_events

logger = logging.getLogger(__name__)

_MISSING_TABLE_ERRORS = (ProgrammingError, OperationalError)
_SQL_HISTORY_OVERFETCH = 8

# Prefer the public alias used by readers and writers.
MISSING_TABLE_ERRORS = _MISSING_TABLE_ERRORS


DEFAULT_EVENTS_TABLE = "events"
DEFAULT_USERS_TABLE = "users"
DEFAULT_ITEMS_TABLE = "items"
DEFAULT_RECOMMENDATIONS_TABLE = "recommendations"
DEFAULT_MANIFEST_TABLE = "recommendation_runs"
DEFAULT_MODEL_ARTIFACT_TABLE = "model_artifacts"
# Separate from DEFAULT_ITEMS_TABLE so shared in/out DBs do not clobber source items.
DEFAULT_RECOMMENDATION_ITEMS_TABLE = "recommendation_items"
DEFAULT_EXPOSURES_TABLE = "recommendation_exposures"
DEFAULT_EXPERIMENT_STATE_TABLE = "experiment_state"

DEFAULT_DB_TABLES = frozenset(
    {
        DEFAULT_EVENTS_TABLE,
        DEFAULT_USERS_TABLE,
        DEFAULT_ITEMS_TABLE,
        DEFAULT_RECOMMENDATIONS_TABLE,
        DEFAULT_MANIFEST_TABLE,
        DEFAULT_MODEL_ARTIFACT_TABLE,
        DEFAULT_RECOMMENDATION_ITEMS_TABLE,
        DEFAULT_EXPOSURES_TABLE,
        DEFAULT_EXPERIMENT_STATE_TABLE,
    }
)


def _clear_table_for_replace(conn, table: str) -> None:
    """Empty ``table`` before a full rewrite; prefer TRUNCATE, fall back to DELETE.

    Some backends (e.g. SQLite) lack TRUNCATE. Rolling back only the TRUNCATE
    savepoint and then appending would duplicate rows, so DELETE is tried next.
    If the table does not exist yet, both fail and ``to_sql`` creates it.
    """
    savepoint = conn.begin_nested()
    try:
        conn.execute(text(f'TRUNCATE TABLE "{table}"'))
        savepoint.commit()
        return
    except _MISSING_TABLE_ERRORS:
        savepoint.rollback()

    savepoint = conn.begin_nested()
    try:
        conn.execute(text(f'DELETE FROM "{table}"'))
        savepoint.commit()
    except _MISSING_TABLE_ERRORS:
        savepoint.rollback()


def _missing_optional_recommendation_columns(engine: Engine, table: str, frame: pd.DataFrame) -> list[str]:
    wanted = [column for column in (REASONS_COLUMN, VARIANT_COLUMN) if column in frame.columns]
    if not wanted:
        return []
    inspector = inspect(engine)
    if not inspector.has_table(table):
        return []
    existing = {column["name"] for column in inspector.get_columns(table)}
    return [column for column in wanted if column not in existing]


def _require_optional_recommendation_columns(engine: Engine, table: str, frame: pd.DataFrame) -> None:
    missing = _missing_optional_recommendation_columns(engine, table, frame)
    if not missing:
        return
    alters = "; ".join(f"ALTER TABLE … ADD COLUMN {column} TEXT" for column in missing)
    raise RecommendationSchemaError(
        f"Recommendations table {table!r} is missing column(s) {missing}; {alters}"
    )


def _sql_user_source(query: str | None, table: str) -> str:
    if not query:
        return f'"{table}"'
    return f"({query.strip().rstrip(';').strip()}) AS _cicerone_user_rows"


class DatabaseInputSource:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def _configured_query(self, key: str) -> str | None:
        query = self._options.get(key)
        if query is None:
            return None
        return readonly_select(query, option=f"input.options.{key}")

    def _read(self, query: str | None, table: str) -> pd.DataFrame:
        sql = query or f'SELECT * FROM "{table}"'
        logger.info("Reading from database: %s", "configured query" if query else f'table "{table}"')
        return pd.read_sql(text(sql), self._engine)

    def read_events(self) -> pd.DataFrame:
        return self._read(
            self._configured_query("events_query"),
            sql_identifier(self._options.get("events_table", DEFAULT_EVENTS_TABLE), option="events_table"),
        )

    def _read_optional(self, query: str | None, table: str, label: str) -> pd.DataFrame | None:
        if query is None and not inspect(self._engine).has_table(table):
            logger.warning(
                "Optional %s source (table %r) does not exist — continuing without %s features.",
                label,
                table,
                label,
            )
            return None
        try:
            return self._read(query, table)
        except _MISSING_TABLE_ERRORS:
            logger.warning("Optional %s source unavailable — continuing without %s features.", label, label)
            return None

    def read_users(self) -> pd.DataFrame | None:
        return self._read_optional(
            self._configured_query("users_query"),
            sql_identifier(self._options.get("users_table", DEFAULT_USERS_TABLE), option="users_table"),
            "users",
        )

    def read_items(self) -> pd.DataFrame | None:
        return self._read_optional(
            self._configured_query("items_query"),
            sql_identifier(self._options.get("items_table", DEFAULT_ITEMS_TABLE), option="items_table"),
            "items",
        )

    def _select_user_rows(
        self,
        query: str | None,
        table: str,
        user_id: str,
        *,
        limit: int | None = None,
        order_occurred_at: bool = False,
    ) -> pd.DataFrame:
        source = _sql_user_source(query, table)
        sql = f'SELECT * FROM {source} WHERE "user_id" = :user_id'
        params: dict[str, Any] = {"user_id": user_id}
        if order_occurred_at:
            sql += f' ORDER BY "{OCCURRED_AT_COLUMN}" DESC NULLS LAST'
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = int(limit)
        logger.info(
            "Reading from database: user-filtered %s",
            "configured query" if query else f'table "{table}"',
        )
        return pd.read_sql(text(sql), self._engine, params=params)

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        table = sql_identifier(self._options.get("events_table", DEFAULT_EVENTS_TABLE), option="events_table")
        query = self._configured_query("events_query")
        sql_limit = max(int(limit) * _SQL_HISTORY_OVERFETCH, int(limit))
        try:
            frame = self._select_user_rows(query, table, user_id, limit=sql_limit, order_occurred_at=True)
        except _MISSING_TABLE_ERRORS:
            frame = self._select_user_rows(query, table, user_id, limit=sql_limit)
        return newest_events(filter_rows_for_user(frame, user_id), limit)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        query = self._configured_query("users_query")
        table = sql_identifier(self._options.get("users_table", DEFAULT_USERS_TABLE), option="users_table")
        if query is None and not inspect(self._engine).has_table(table):
            return None
        try:
            frame = self._select_user_rows(query, table, user_id)
        except _MISSING_TABLE_ERRORS:
            return None
        matched = filter_rows_for_user(frame, user_id)
        if matched.empty:
            return None
        return matched.iloc[0].to_dict()


class DatabaseOutputSink:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def write_recommendations(self, df: pd.DataFrame) -> None:
        table, _columns, _user_col = recommendations_sql_names(
            self._options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
        )
        logger.info("Writing %d rows to database table %r", len(df), table)
        _require_optional_recommendation_columns(self._engine, table, df)
        with self._engine.begin() as conn:
            _clear_table_for_replace(conn, table)
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)

    def replace_recommendations_for_users(self, df: pd.DataFrame, *, user_ids: Sequence[str]) -> int:
        ids = normalize_replace_user_ids(df, user_ids)
        if not ids:
            return 0
        table, _columns, user_col = recommendations_sql_names(
            self._options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
        )
        logger.info(
            "Replacing recommendations for %d user(s) (%d row(s)) in %r",
            len(ids),
            len(df),
            table,
        )
        _require_optional_recommendation_columns(self._engine, table, df)
        # Identifiers are sql_identifier-validated; match events.store SELECT quoting.
        delete_sql = text(f"DELETE FROM {table} WHERE {user_col} IN :user_ids").bindparams(
            bindparam("user_ids", expanding=True)
        )
        count_sql = text(f"SELECT COUNT(DISTINCT {user_col}) FROM {table}")
        with self._engine.begin() as conn:
            savepoint = conn.begin_nested()
            try:
                conn.execute(delete_sql, {"user_ids": ids})
                savepoint.commit()
            except _MISSING_TABLE_ERRORS as exc:
                savepoint.rollback()
                logger.warning(
                    "Recommendations delete skipped for table %r (missing table or schema issue): %s",
                    table,
                    exc,
                )
                # Schema drift (e.g. missing user_id): do not append into a broken table.
                if is_missing_column_error(exc):
                    raise RecommendationSchemaError(
                        f"Recommendations schema mismatch for table {table!r}; refusing replace"
                    ) from exc
            if not df.empty:
                df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
            count_savepoint = conn.begin_nested()
            try:
                value = conn.execute(count_sql).scalar()
                count_savepoint.commit()
            except _MISSING_TABLE_ERRORS as exc:
                count_savepoint.rollback()
                logger.warning(
                    "Recommendations count skipped for table %r (missing table or schema issue): %s",
                    table,
                    exc,
                )
                return 0
        return int(value or 0)

    def write_manifest(self, manifest: dict) -> None:
        table = sql_identifier(
            self._options.get("manifest_table", DEFAULT_MANIFEST_TABLE),
            option="manifest_table",
        )
        logger.info("Appending run manifest to database table %r", table)
        pd.DataFrame([manifest]).to_sql(table, self._engine, if_exists="append", index=False)

    def write_model_artifact(self, payload: bytes) -> None:
        """Replace the single-row model_artifacts table with the latest blob."""
        table_name = sql_identifier(
            self._options.get("model_artifact_table", DEFAULT_MODEL_ARTIFACT_TABLE),
            option="model_artifact_table",
        )
        logger.info("Writing model artifact (%d bytes) to database table %r", len(payload), table_name)
        metadata = MetaData()
        artifacts = Table(
            table_name,
            metadata,
            Column("payload", LargeBinary, nullable=False),
            Column("written_at", DateTime(timezone=True), nullable=False),
        )
        with self._engine.begin() as conn:
            artifacts.create(conn, checkfirst=True)
            conn.execute(artifacts.delete())
            conn.execute(insert(artifacts).values(payload=payload, written_at=datetime.now(UTC)))

    def replace_model_artifact_if(self, payload: bytes, expected_fingerprint: str) -> bool:
        table_name = sql_identifier(
            self._options.get("model_artifact_table", DEFAULT_MODEL_ARTIFACT_TABLE),
            option="model_artifact_table",
        )
        metadata = MetaData()
        artifacts = Table(
            table_name,
            metadata,
            Column("payload", LargeBinary, nullable=False),
            Column("written_at", DateTime(timezone=True), nullable=False),
        )
        with self._engine.begin() as conn:
            artifacts.create(conn, checkfirst=True)
            row = conn.execute(select(artifacts.c.written_at).limit(1).with_for_update()).first()
            if row is None or row[0] is None:
                return False
            written = row[0]
            stamp = written.isoformat() if hasattr(written, "isoformat") else str(written)
            if f"db:{stamp}" != expected_fingerprint:
                return False
            deleted = conn.execute(artifacts.delete().where(artifacts.c.written_at == written))
            if deleted.rowcount < 1:
                return False
            conn.execute(insert(artifacts).values(payload=payload, written_at=datetime.now(UTC)))
        return True

    def read_model_artifact(self) -> bytes | None:
        table_name = sql_identifier(
            self._options.get("model_artifact_table", DEFAULT_MODEL_ARTIFACT_TABLE),
            option="model_artifact_table",
        )
        if not inspect(self._engine).has_table(table_name):
            return None
        with self._engine.connect() as conn:
            row = conn.execute(text(f'SELECT payload FROM "{table_name}" LIMIT 1')).first()
        if row is None or row[0] is None:
            return None
        return bytes(row[0])

    def model_artifact_fingerprint(self) -> str | None:
        table_name = sql_identifier(
            self._options.get("model_artifact_table", DEFAULT_MODEL_ARTIFACT_TABLE),
            option="model_artifact_table",
        )
        if not inspect(self._engine).has_table(table_name):
            return None
        with self._engine.connect() as conn:
            row = conn.execute(text(f'SELECT written_at FROM "{table_name}" LIMIT 1')).first()
        if row is None or row[0] is None:
            return None
        written = row[0]
        stamp = written.isoformat() if hasattr(written, "isoformat") else str(written)
        return f"db:{stamp}"

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        table = sql_identifier(
            self._options.get("recommendation_items_table", DEFAULT_RECOMMENDATION_ITEMS_TABLE),
            option="recommendation_items_table",
        )
        logger.info("Writing %d item snapshot rows to database table %r", len(df), table)
        with self._engine.begin() as conn:
            _clear_table_for_replace(conn, table)
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
