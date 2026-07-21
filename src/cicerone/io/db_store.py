"""Database input/output backend (SQLAlchemy). Lets the job read straight
from a Postgres source (e.g. a torbido read-replica, via a table name or a
custom SQL query) and/or write recommendations into a Postgres table that
an application can query directly.

Table/column identifiers used here come from trusted deploy-time
configuration (environment variables), never from end-user input.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from cicerone.config import DatabaseLocation

logger = logging.getLogger(__name__)


class DatabaseInputSource:
    def __init__(self, location: DatabaseLocation):
        self._loc = location
        self._engine = create_engine(location.url, pool_pre_ping=True)

    def _read(self, query: str | None, table: str) -> pd.DataFrame:
        sql = query or f'SELECT * FROM "{table}"'
        logger.info("Reading from database: %s", sql if query else f'table "{table}"')
        return pd.read_sql(text(sql), self._engine)

    def read_events(self) -> pd.DataFrame:
        return self._read(self._loc.events_query, self._loc.events_table)

    def read_users(self) -> pd.DataFrame | None:
        try:
            return self._read(self._loc.users_query, self._loc.users_table)
        except Exception:  # noqa: BLE001 - optional input, missing table/query is expected
            logger.warning("Optional users source unavailable — continuing without user features.")
            return None

    def read_items(self) -> pd.DataFrame | None:
        try:
            return self._read(self._loc.items_query, self._loc.items_table)
        except Exception:  # noqa: BLE001 - optional input, missing table/query is expected
            logger.warning("Optional items source unavailable — continuing without item features.")
            return None


class DatabaseOutputSink:
    def __init__(self, location: DatabaseLocation):
        self._loc = location
        self._engine = create_engine(location.url, pool_pre_ping=True)

    def write_recommendations(self, df: pd.DataFrame) -> None:
        table = self._loc.recommendations_table
        logger.info("Writing %d rows to database table %r", len(df), table)
        with self._engine.begin() as conn:
            # Replace the previous "latest" snapshot. TRUNCATE is wrapped in
            # a savepoint so a first-ever run (table doesn't exist yet) just
            # falls through to to_sql() creating it.
            savepoint = conn.begin_nested()
            try:
                conn.execute(text(f'TRUNCATE TABLE "{table}"'))
                savepoint.commit()
            except ProgrammingError:
                savepoint.rollback()
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)

    def write_manifest(self, manifest: dict) -> None:
        table = self._loc.manifest_table
        logger.info("Appending run manifest to database table %r", table)
        pd.DataFrame([manifest]).to_sql(table, self._engine, if_exists="append", index=False)
