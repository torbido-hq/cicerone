"""Shared helpers for Postgres-backed system / DB tests.

Keeps schema-reset guardrails and fixture normalization out of the
end-to-end scenario module so they stay reusable and unit-testable.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine

from cicerone.io.db_store import DEFAULT_DB_TABLES
from support.postgres_defaults import looks_like_test_database, postgres_test_db


def is_dedicated_test_database(db_name: str | None) -> bool:
    """True when ``db_name`` is safe for destructive test schema resets.

    Pattern-only: do not trust ``postgres_test_db()`` / ``POSTGRES_TEST_DB``
    env, which could be overridden to the app DB name (e.g. ``cicerone``).
    """
    return looks_like_test_database(db_name)


def reset_schema(engine: Engine) -> None:
    """Drop known Cicerone tables in the connected database.

    Reflects the schema, then drops only tables in ``DEFAULT_DB_TABLES``
    that currently exist — never an unrelated table that happens to share
    the DB. Guarded: only dedicated test DB names, and only when
    ``ALLOW_SCHEMA_RESET_FOR_TESTS=1`` (see CONTRIBUTING.md).
    """
    db_name = engine.url.database
    if not is_dedicated_test_database(db_name):
        raise RuntimeError(
            f"Refusing to reset schema for non-test database {db_name!r}. "
            "TEST_DATABASE_URL must point at a dedicated test DB "
            f"(e.g. {postgres_test_db()!r}, or a name starting with 'test_' / "
            "ending with '_test')."
        )
    if os.environ.get("ALLOW_SCHEMA_RESET_FOR_TESTS") != "1":
        raise RuntimeError(
            "Schema reset for tests is disabled. Set ALLOW_SCHEMA_RESET_FOR_TESTS=1 "
            "to permit dropping known Cicerone tables on the dedicated test database."
        )

    metadata = MetaData()
    metadata.reflect(bind=engine)
    for table_name in list(metadata.tables):
        if table_name not in DEFAULT_DB_TABLES:
            metadata.remove(metadata.tables[table_name])
    metadata.drop_all(bind=engine)


def postgres_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a fixture frame into a shape psycopg can insert (lists, not ndarrays)."""
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].map(
            lambda value: (
                value.tolist()
                if hasattr(value, "tolist")
                else (list(value) if isinstance(value, tuple) else value)
            )
        )
    return out
