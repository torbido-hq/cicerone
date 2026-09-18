"""Postgres-specific helpers for system / DB tests.

Schema-reset guardrails and fixture normalization stay here. Shared catalog,
TOML, and HTTP mounts live in ``support.system_spec`` and are re-exported.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine
from support.postgres_defaults import canonical_postgres_test_db, looks_like_test_database
from support.system_spec import (
    REPO_FEATURES_CONFIG,
    REPO_ROOT,
    SYSTEM_DASHBOARD_PASSWORD,
    SYSTEM_DASHBOARD_USER,
    SYSTEM_SERVE_TOKEN,
    available_recommendation_ids,
    dashboard_users,
    mount_dashboard_app,
    mount_serve_app,
    run_system_job,
    sample_system_catalog,
    write_system_config,
)

from cicerone.io.db_store import (
    DEFAULT_DB_TABLES,
    DEFAULT_EVENTS_TABLE,
    DEFAULT_ITEMS_TABLE,
    DEFAULT_USERS_TABLE,
)

__all__ = [
    "REPO_FEATURES_CONFIG",
    "REPO_ROOT",
    "SYSTEM_DASHBOARD_PASSWORD",
    "SYSTEM_DASHBOARD_USER",
    "SYSTEM_SERVE_TOKEN",
    "available_recommendation_ids",
    "dashboard_users",
    "is_dedicated_test_database",
    "mount_dashboard_app",
    "mount_serve_app",
    "postgres_ready",
    "reset_schema",
    "run_system_job",
    "sample_system_catalog",
    "seed_catalog",
    "write_system_config",
]


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
        # Prefer canonical_postgres_test_db() — postgres_test_db() can raise on hostile overrides.
        raise RuntimeError(
            f"Refusing to reset schema for non-test database {db_name!r}. "
            "TEST_DATABASE_URL must point at a dedicated test DB "
            f"(e.g. {canonical_postgres_test_db()!r}, or a name starting with "
            "'test_' / ending with '_test')."
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


def seed_catalog(engine: Engine, events: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> None:
    """Persist catalog frames via the same table names the db input source reads."""
    postgres_ready(events).to_sql(DEFAULT_EVENTS_TABLE, engine, if_exists="replace", index=False)
    postgres_ready(users).to_sql(DEFAULT_USERS_TABLE, engine, if_exists="replace", index=False)
    postgres_ready(items).to_sql(DEFAULT_ITEMS_TABLE, engine, if_exists="replace", index=False)
