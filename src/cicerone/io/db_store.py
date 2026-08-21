"""Database input/output via SQLAlchemy.

Options: database_url (required); optional table names / raw SQL overrides
(events_query, users_query, items_query). Identifiers come from trusted
deploy-time config only.
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
    LargeBinary,
    MetaData,
    Table,
    bindparam,
    create_engine,
    insert,
    inspect,
    text,
)
from sqlalchemy.exc import OperationalError, ProgrammingError

from cicerone.io.db_errors import is_missing_column_error
from cicerone.io.options import require_option, sql_identifier
from cicerone.io.recommendation_schema import recommendations_sql_names
from cicerone.io.replace_users import RecommendationSchemaError, normalize_replace_user_ids

logger = logging.getLogger(__name__)

_MISSING_TABLE_ERRORS = (ProgrammingError, OperationalError)

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

DEFAULT_DB_TABLES = frozenset(
    {
        DEFAULT_EVENTS_TABLE,
        DEFAULT_USERS_TABLE,
        DEFAULT_ITEMS_TABLE,
        DEFAULT_RECOMMENDATIONS_TABLE,
        DEFAULT_MANIFEST_TABLE,
        DEFAULT_MODEL_ARTIFACT_TABLE,
        DEFAULT_RECOMMENDATION_ITEMS_TABLE,
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


class DatabaseInputSource:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def _read(self, query: str | None, table: str) -> pd.DataFrame:
        sql = query or f'SELECT * FROM "{table}"'
        logger.info("Reading from database: %s", sql if query else f'table "{table}"')
        return pd.read_sql(text(sql), self._engine)

    def read_events(self) -> pd.DataFrame:
        return self._read(
            self._options.get("events_query"),
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
            self._options.get("users_query"),
            sql_identifier(self._options.get("users_table", DEFAULT_USERS_TABLE), option="users_table"),
            "users",
        )

    def read_items(self) -> pd.DataFrame | None:
        return self._read_optional(
            self._options.get("items_query"),
            sql_identifier(self._options.get("items_table", DEFAULT_ITEMS_TABLE), option="items_table"),
            "items",
        )


class DatabaseOutputSink:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def write_recommendations(self, df: pd.DataFrame) -> None:
        table, _columns, _user_col = recommendations_sql_names(
            self._options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
        )
        logger.info("Writing %d rows to database table %r", len(df), table)
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

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        table = sql_identifier(
            self._options.get("recommendation_items_table", DEFAULT_RECOMMENDATION_ITEMS_TABLE),
            option="recommendation_items_table",
        )
        logger.info("Writing %d item snapshot rows to database table %r", len(df), table)
        with self._engine.begin() as conn:
            _clear_table_for_replace(conn, table)
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
