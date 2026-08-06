"""Database input/output via SQLAlchemy.

Options: database_url (required); optional table names / raw SQL overrides
(events_query, users_query, items_query). Identifiers come from trusted
deploy-time config only.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Column, DateTime, LargeBinary, MetaData, Table, create_engine, insert, inspect, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from cicerone.io.options import require_option, sql_identifier

logger = logging.getLogger(__name__)

_MISSING_TABLE_ERRORS = (ProgrammingError, OperationalError)

DEFAULT_EVENTS_TABLE = "events"
DEFAULT_USERS_TABLE = "users"
DEFAULT_ITEMS_TABLE = "items"
DEFAULT_RECOMMENDATIONS_TABLE = "recommendations"
DEFAULT_MANIFEST_TABLE = "recommendation_runs"
DEFAULT_MODEL_ARTIFACT_TABLE = "model_artifacts"
# Snapshot of items written next to recommendations for serve-time filters
# (category / availability). Kept separate from DEFAULT_ITEMS_TABLE so a
# shared input+output database cannot clobber the source items table.
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
        table = sql_identifier(
            self._options.get("recommendations_table", DEFAULT_RECOMMENDATIONS_TABLE),
            option="recommendations_table",
        )
        logger.info("Writing %d rows to database table %r", len(df), table)
        with self._engine.begin() as conn:
            _clear_table_for_replace(conn, table)
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)

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
