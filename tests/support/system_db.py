"""Shared helpers for Postgres-backed system / DB tests.

Keeps schema-reset guardrails and fixture normalization out of the
end-to-end scenario module so they stay reusable and unit-testable.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine

from cicerone.io.db_store import (
    DEFAULT_DB_TABLES,
    DEFAULT_EVENTS_TABLE,
    DEFAULT_ITEMS_TABLE,
    DEFAULT_USERS_TABLE,
)
from support.postgres_defaults import canonical_postgres_test_db, looks_like_test_database

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_FEATURES_CONFIG = REPO_ROOT / "config" / "features.toml"
SYSTEM_SERVE_TOKEN = "system-spec-secret"
SYSTEM_DASHBOARD_USER = "alice"
SYSTEM_DASHBOARD_PASSWORD = "s3cret"


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


def sample_system_catalog() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shared events/users/items used by the Postgres system spec."""
    now = pd.Timestamp("2026-09-01T12:00:00Z")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 3, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i2", "event_type": "cart_add", "quantity": 1, "occurred_at": now},
        ]
    )
    users = pd.DataFrame(
        [
            {"user_id": "u1", "favorite_styles": ["ipa", "stout"], "region_slug": "lazio"},
            {"user_id": "u2", "favorite_styles": ["lager"], "region_slug": "toscana"},
            {"user_id": "u3", "favorite_styles": [], "region_slug": None},
            {"user_id": "u4", "favorite_styles": ["ipa"], "region_slug": "lazio"},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": False},
            {"item_id": "i4", "category": "wine", "producer_id": "p3", "published": False, "in_stock": True},
        ]
    )
    return events, users, items


def seed_catalog(engine: Engine, events: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> None:
    """Persist catalog frames via the same table names the db input source reads."""
    postgres_ready(events).to_sql(DEFAULT_EVENTS_TABLE, engine, if_exists="replace", index=False)
    postgres_ready(users).to_sql(DEFAULT_USERS_TABLE, engine, if_exists="replace", index=False)
    postgres_ready(items).to_sql(DEFAULT_ITEMS_TABLE, engine, if_exists="replace", index=False)


def write_system_config(
    path: Path,
    *,
    database_url: str,
    feature_config_path: Path | str = REPO_FEATURES_CONFIG,
    serve_token: str = SYSTEM_SERVE_TOKEN,
) -> Path:
    """Write the shared system-spec TOML (db I/O, artifact, serve, dashboard, track, eval)."""
    path.write_text(
        f"""
        [job]
        top_k = 3
        feature_config_path = "{feature_config_path}"
        models = ["collaborative", "popular"]
        save_model_artifact = true

        [job.eval]
        enabled = true

        [input]
        kind = "db"
        [input.options]
        database_url = "{database_url}"

        [output]
        kind = "db"
        [output.options]
        database_url = "{database_url}"

        [serve]
        auth_token = "{serve_token}"
        category_column = "category"
        default_k = 3

        [dashboard]
        enabled = true

        [track]
        enabled = true
        """
    )
    return path


def available_recommendation_ids(
    recs: pd.DataFrame,
    items: pd.DataFrame | None,
    *,
    availability_filters: Sequence[str],
    category_column: str = "category",
    k: int | None = None,
) -> list[str]:
    """Item ids after the same availability filter serve applies by default."""
    from cicerone.serve.item_filters import available_item_ids, filter_recommendations

    filtered = filter_recommendations(
        recs,
        items=items,
        available_ids=available_item_ids(items, availability_filters) if items is not None else None,
        category=None,
        category_column=category_column,
        exclude_unavailable=True,
    )
    ids = list(filtered["item_id"].astype(str))
    return ids if k is None else ids[:k]


def dashboard_users(
    username: str = SYSTEM_DASHBOARD_USER,
    password: str = SYSTEM_DASHBOARD_PASSWORD,
) -> dict[str, str]:
    import bcrypt

    return {username: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")}


def mount_serve_app(settings: Any):
    from cicerone.feature_config import load_feature_config
    from cicerone.io.factory import build_manifest_reader, build_recommendation_reader
    from cicerone.serve import create_app
    from cicerone.serve.item_filters import configure_reader_item_filters

    reader = build_recommendation_reader(settings.output)
    feature_path = Path(settings.feature_config_path)
    feature_config = load_feature_config(feature_path) if feature_path.is_file() else None
    availability = list(feature_config.item_availability_filters) if feature_config else []
    configure_reader_item_filters(
        reader,
        category_column=settings.serve.category_column,
        availability_filters=availability,
    )
    return create_app(
        settings,
        reader,
        manifest_reader=build_manifest_reader(settings.output),
        feature_config=feature_config,
    )


def mount_dashboard_app(
    settings: Any,
    users: dict[str, str],
    *,
    config_path: str | Path | None = None,
):
    from cicerone.dashboard import create_app
    from cicerone.io.factory import (
        build_manifest_reader,
        build_recommendation_reader,
        build_user_history_reader,
    )

    return create_app(
        settings,
        build_manifest_reader(settings.output),
        users,
        build_recommendation_reader(settings.output),
        build_user_history_reader(settings.input),
        config_path=None if config_path is None else str(config_path),
    )
