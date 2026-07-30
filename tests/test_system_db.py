"""System-style end-to-end check against a real Postgres (Rails system-spec analogue).

Seeds events/users/items, runs the full batch job with db input/output + a
model artifact, then verifies what serve/dashboard would read back — all
through the same SQLAlchemy stores production uses.

Requires TEST_DATABASE_URL (set automatically by docker-compose.ci.yml, or
locally via the compose ``postgres`` service — see CONTRIBUTING.md).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from cicerone import job
from cicerone.artifact import ARTIFACT_SCHEMA_VERSION, loads_artifact, recommend_from_artifact
from cicerone.io.manifest_reader import DbManifestReader
from cicerone.io.recommendation_reader import DbRecommendationReader

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
REPO_FEATURES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "features.toml"

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set — start compose postgres "
    "(`docker compose up -d postgres`) and export "
    "TEST_DATABASE_URL=postgresql+psycopg://cicerone:cicerone@localhost:5432/cicerone, "
    "or run via docker-compose.ci.yml",
)

_TABLES = (
    "events",
    "users",
    "items",
    "recommendations",
    "recommendation_runs",
    "model_artifacts",
)


@pytest.fixture(autouse=True)
def _clean_tables():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        for table in _TABLES:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
    yield
    with engine.begin() as conn:
        for table in _TABLES:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
    engine.dispose()


def _seed_sample_catalog(engine) -> None:
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


def test_system_job_db_round_trip_with_artifact_and_readers(tmp_path, monkeypatch):
    """Full path: Postgres catalog → job.run → recommendations/manifest/artifact
    → recommendation + manifest readers (serve/dashboard backends).
    """
    engine = create_engine(TEST_DATABASE_URL)
    _seed_sample_catalog(engine)

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

    recommendations = pd.read_sql('SELECT * FROM "recommendations"', engine)
    assert set(recommendations["user_id"]) >= {"u1", "u2", "u3"}
    assert set(recommendations.columns) >= {"user_id", "item_id", "rank", "score", "source"}
    assert recommendations["rank"].min() >= 1

    manifests = pd.read_sql(
        'SELECT * FROM "recommendation_runs" ORDER BY generated_at DESC',
        engine,
    )
    assert len(manifests) == 1
    manifest = manifests.iloc[0].to_dict()
    assert manifest["status"] == "success"
    assert manifest["triggered_by"] == "system-spec"
    assert int(manifest["n_events"]) == 5
    assert bool(manifest["artifact_written"]) is True
    assert int(manifest["artifact_schema_version"]) == ARTIFACT_SCHEMA_VERSION

    artifacts = pd.read_sql('SELECT payload, written_at FROM "model_artifacts"', engine)
    assert len(artifacts) == 1
    payload = bytes(artifacts.iloc[0]["payload"])
    loaded = loads_artifact(payload)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert "collaborative" in loaded.models or "popular" in loaded.models

    from_artifact = recommend_from_artifact(loaded, ["u1", "u2"], top_k=3)
    assert not from_artifact.empty
    assert set(from_artifact["user_id"]) <= {"u1", "u2"}

    # Serve-path reader (same store the HTTP API would query).
    rec_reader = DbRecommendationReader({"database_url": TEST_DATABASE_URL})
    served = rec_reader.get_recommendations("u1", k=2)
    assert len(served) == 2
    assert set(served["user_id"]) == {"u1"}
    # Ordered by rank ascending (priority combine may leave equal ranks
    # across strategies for different items).
    assert list(served["rank"]) == sorted(served["rank"].tolist())
    assert set(served.columns) >= {"user_id", "item_id", "rank", "score", "source"}

    # Dashboard-path reader.
    manifest_reader = DbManifestReader({"database_url": TEST_DATABASE_URL})
    latest = manifest_reader.read_latest()
    assert latest is not None
    assert latest["status"] == "success"
    assert latest["triggered_by"] == "system-spec"
    recent = manifest_reader.read_recent(limit=5)
    assert len(recent) == 1

    engine.dispose()
