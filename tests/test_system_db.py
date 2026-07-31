"""System-style end-to-end check against a real Postgres (Rails system-spec analogue).

Seeds events/users/items from the shared conftest fixtures, runs the full
batch job with db input/output + a model artifact, then verifies what
serve/dashboard would read back — through the same SQLAlchemy stores
production uses.

Requires a test DB URL via ``TEST_DATABASE_URL`` or ``POSTGRES_TEST_HOST``
(see ``support.postgres_defaults`` / CONTRIBUTING.md). Schema resets are
gated by ``support.system_db.reset_schema``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from support.postgres_defaults import resolve_test_database_url
from support.system_db import postgres_ready, reset_schema

from cicerone import job
from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, loads_artifact, recommend_from_artifact
from cicerone.io.db_store import (
    DEFAULT_EVENTS_TABLE,
    DEFAULT_ITEMS_TABLE,
    DEFAULT_MODEL_ARTIFACT_TABLE,
    DEFAULT_USERS_TABLE,
)
from cicerone.io.manifest_reader import DbManifestReader
from cicerone.io.recommendation_reader import DbRecommendationReader

TEST_DATABASE_URL = resolve_test_database_url()
REPO_FEATURES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "features.toml"

_SKIP_NO_TEST_DB = (
    "TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — start compose postgres "
    "(`docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres`) "
    "and export POSTGRES_TEST_HOST=localhost ALLOW_SCHEMA_RESET_FOR_TESTS=1, "
    "or run via docker-compose.ci.yml"
)


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """One Engine for the whole test session — avoids per-test connect/dispose."""
    if not TEST_DATABASE_URL:
        pytest.skip(_SKIP_NO_TEST_DB)
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def clean_schema(db_engine: Engine) -> Iterator[None]:
    """Reset schema once around the system-spec."""
    reset_schema(db_engine)
    yield
    reset_schema(db_engine)


def _seed_catalog(engine: Engine, events: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> None:
    """Persist the shared sample fixtures via the same table names the db input source reads."""
    postgres_ready(events).to_sql(DEFAULT_EVENTS_TABLE, engine, if_exists="replace", index=False)
    postgres_ready(users).to_sql(DEFAULT_USERS_TABLE, engine, if_exists="replace", index=False)
    postgres_ready(items).to_sql(DEFAULT_ITEMS_TABLE, engine, if_exists="replace", index=False)


@pytest.mark.skipif(not TEST_DATABASE_URL, reason=_SKIP_NO_TEST_DB)
def test_system_job_db_round_trip_with_artifact_and_readers(
    db_engine: Engine,
    clean_schema: None,
    tmp_path,
    monkeypatch,
    sample_events: pd.DataFrame,
    sample_users: pd.DataFrame,
    sample_items: pd.DataFrame,
):
    """Full path: Postgres catalog → job.run → recommendations/manifest/artifact
    → recommendation + manifest readers (serve/dashboard backends).
    """
    _seed_catalog(db_engine, sample_events, sample_users, sample_items)

    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        top_k = 3
        feature_config_path = "{REPO_FEATURES_CONFIG}"
        models = ["collaborative", "popular"]
        save_model_artifact = true

        [input]
        kind = "db"
        [input.options]
        database_url = "{TEST_DATABASE_URL}"

        [output]
        kind = "db"
        [output.options]
        database_url = "{TEST_DATABASE_URL}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    job.run(triggered_by="system-spec")

    expected_users = set(sample_events["user_id"]) | set(sample_users["user_id"])

    # Serve-path reader (same store the HTTP API would query).
    rec_reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    for user_id in sorted(expected_users):
        served_all = rec_reader.get_recommendations(user_id, k=10)
        assert not served_all.empty, f"expected recommendations for {user_id}"
        assert set(served_all.columns) >= {"user_id", "item_id", "rank", "score", "source"}
        assert served_all["rank"].min() >= 1

    served = rec_reader.get_recommendations("u1", k=2)
    assert len(served) == 2
    assert set(served["user_id"]) == {"u1"}
    # Ordered by rank ascending (priority combine may leave equal ranks
    # across strategies for different items).
    assert list(served["rank"]) == sorted(served["rank"].tolist())

    # Dashboard-path reader.
    manifest_reader = DbManifestReader({"database_url": TEST_DATABASE_URL})
    latest = manifest_reader.read_latest()
    assert latest is not None
    assert latest["status"] == "success"
    assert latest["triggered_by"] == "system-spec"
    assert int(latest["n_events"]) == len(sample_events)
    assert bool(latest["artifact_written"]) is True
    assert int(latest["artifact_schema_version"]) == ARTIFACT_SCHEMA_VERSION
    recent = manifest_reader.read_recent(limit=5)
    assert len(recent) == 1

    # Artifact blob has no dedicated reader — load via the public artifact API.
    artifacts = pd.read_sql(
        text(f'SELECT payload FROM "{DEFAULT_MODEL_ARTIFACT_TABLE}"'),
        db_engine,
    )
    assert len(artifacts) == 1
    payload = bytes(artifacts.iloc[0]["payload"])
    loaded = loads_artifact(payload)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert "collaborative" in loaded.models or "popular" in loaded.models

    from_artifact = recommend_from_artifact(loaded, ["u1", "u2"], top_k=3)
    assert not from_artifact.empty
    assert set(from_artifact["user_id"]) <= {"u1", "u2"}
