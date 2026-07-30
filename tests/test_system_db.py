"""System-style end-to-end check against a real Postgres (Rails system-spec analogue).

Seeds events/users/items, runs the full batch job with db input/output + a
model artifact, then verifies what serve/dashboard would read back — all
through the same SQLAlchemy stores production uses.

Requires TEST_DATABASE_URL (set automatically by docker-compose.ci.yml, or
locally via the compose ``postgres`` service's ``cicerone_test`` database —
see CONTRIBUTING.md). Prefer a dedicated test database; this module drops
every table in that database's public schema between runs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import Engine

from cicerone import job
from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, loads_artifact, recommend_from_artifact
from cicerone.io.manifest_reader import DbManifestReader
from cicerone.io.recommendation_reader import DbRecommendationReader

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
REPO_FEATURES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "features.toml"

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set — start compose postgres "
    "(`docker compose --profile db up -d postgres`) and export "
    "TEST_DATABASE_URL=postgresql+psycopg://cicerone:cicerone@localhost:5432/cicerone_test, "
    "or run via docker-compose.ci.yml",
)


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """One Engine for the whole test session — avoids per-test connect/dispose."""
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


def _reset_schema(engine: Engine) -> None:
    """Drop every reflected table in the connected database.

    Uses SQLAlchemy metadata reflection rather than a hardcoded table list, so
    new job/output tables (or renamed ones) are cleaned up automatically.
    Intended for a dedicated test database only (see CONTRIBUTING.md).
    """
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _clean_schema(db_engine: Engine) -> Iterator[None]:
    _reset_schema(db_engine)
    yield
    _reset_schema(db_engine)


def _seed_sample_catalog(engine: Engine) -> None:
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 2, "occurred_at": now},
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
    # Keep user/item columns scalar so pandas→Postgres doesn't need array
    # adapters; missing feature columns are skipped by dataset.build_dataset.
    users = pd.DataFrame(
        [
            {"user_id": "u1", "region_slug": "lazio"},
            {"user_id": "u2", "region_slug": "toscana"},
            {"user_id": "u3", "region_slug": None},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
        ]
    )
    events.to_sql("events", engine, if_exists="replace", index=False)
    users.to_sql("users", engine, if_exists="replace", index=False)
    items.to_sql("items", engine, if_exists="replace", index=False)


def test_system_job_db_round_trip_with_artifact_and_readers(db_engine: Engine, tmp_path, monkeypatch):
    """Full path: Postgres catalog → job.run → recommendations/manifest/artifact
    → recommendation + manifest readers (serve/dashboard backends).
    """
    _seed_sample_catalog(db_engine)

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

    # Serve-path reader (same store the HTTP API would query).
    rec_reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    for user_id in ("u1", "u2", "u3"):
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
    assert int(latest["n_events"]) == 5
    assert bool(latest["artifact_written"]) is True
    assert int(latest["artifact_schema_version"]) == ARTIFACT_SCHEMA_VERSION
    recent = manifest_reader.read_recent(limit=5)
    assert len(recent) == 1

    # Artifact blob has no dedicated reader — load via the public artifact API.
    artifacts = pd.read_sql(text('SELECT payload FROM "model_artifacts"'), db_engine)
    assert len(artifacts) == 1
    payload = bytes(artifacts.iloc[0]["payload"])
    loaded = loads_artifact(payload)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert "collaborative" in loaded.models or "popular" in loaded.models

    from_artifact = recommend_from_artifact(loaded, ["u1", "u2"], top_k=3)
    assert not from_artifact.empty
    assert set(from_artifact["user_id"]) <= {"u1", "u2"}
