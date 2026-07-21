"""Database input/output backend (SQLAlchemy). Lets the job read straight
from any relational source (e.g. a read-replica, via a table name or a
custom SQL query) and/or write recommendations into a database table that
an application can query directly.

Configured generically via an "options" dict (see cicerone.config.IOSettings)
built from the [input.options] / [output.options] tables in cicerone.toml:

  database_url             required (SQLAlchemy connection string)
  events_table              optional, default "events"
  users_table               optional, default "users"
  items_table                optional, default "items"
  recommendations_table      optional, default "recommendations"
  manifest_table              optional, default "recommendation_runs"
  events_query / users_query / items_query   optional raw SQL overrides —
    use these to read straight from an application's own schema instead of
    requiring it to materialize events/users/items tables verbatim (e.g.
    JOIN orders+order_items+reviews into one query).

Table/column identifiers used here come from trusted deploy-time
configuration, never from end-user input.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from cicerone.io.options import require_option

logger = logging.getLogger(__name__)


class DatabaseInputSource:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def _read(self, query: str | None, table: str) -> pd.DataFrame:
        sql = query or f'SELECT * FROM "{table}"'
        logger.info("Reading from database: %s", sql if query else f'table "{table}"')
        return pd.read_sql(text(sql), self._engine)

    def read_events(self) -> pd.DataFrame:
        return self._read(self._options.get("events_query"), self._options.get("events_table", "events"))

    def read_users(self) -> pd.DataFrame | None:
        try:
            return self._read(self._options.get("users_query"), self._options.get("users_table", "users"))
        except ProgrammingError:
            # Postgres raises ProgrammingError (UndefinedTable) for a missing
            # relation — that's the expected "optional input not configured"
            # case. Anything else (bad credentials, connection errors, ...)
            # propagates so real failures aren't masked.
            logger.warning("Optional users source unavailable — continuing without user features.")
            return None

    def read_items(self) -> pd.DataFrame | None:
        try:
            return self._read(self._options.get("items_query"), self._options.get("items_table", "items"))
        except ProgrammingError:
            logger.warning("Optional items source unavailable — continuing without item features.")
            return None


class DatabaseOutputSink:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def write_recommendations(self, df: pd.DataFrame) -> None:
        table = self._options.get("recommendations_table", "recommendations")
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
        table = self._options.get("manifest_table", "recommendation_runs")
        logger.info("Appending run manifest to database table %r", table)
        pd.DataFrame([manifest]).to_sql(table, self._engine, if_exists="append", index=False)
