from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from cicerone.io.db_store import DatabaseOutputSink
from cicerone.io.manifest_reader import DbManifestReader

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL not set — DB-backed tests run against a real Postgres in CI "
    "(see docker-compose.ci.yml). Set TEST_DATABASE_URL locally to run them.",
)


@pytest.fixture(autouse=True)
def _clean_manifest_table():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS "recommendation_runs"'))
    yield
    engine.dispose()


def test_db_reader_read_latest_returns_the_most_recent_run():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    sink.write_manifest({"generated_at": "2026-07-27T00:00:00+00:00", "status": "success", "error": None})
    sink.write_manifest({"generated_at": "2026-07-28T00:00:00+00:00", "status": "failed", "error": "boom"})

    reader = DbManifestReader({"database_url": TEST_DATABASE_URL})

    latest = reader.read_latest()
    assert latest["status"] == "failed"
    assert latest["error"] == "boom"


def test_db_reader_read_recent_returns_newest_first_up_to_limit():
    sink = DatabaseOutputSink({"database_url": TEST_DATABASE_URL})
    for day in range(1, 4):
        sink.write_manifest({"generated_at": f"2026-07-2{day}T00:00:00+00:00", "status": "success"})

    reader = DbManifestReader({"database_url": TEST_DATABASE_URL})

    runs = reader.read_recent(2)
    assert [run["generated_at"] for run in runs] == [
        "2026-07-23T00:00:00+00:00",
        "2026-07-22T00:00:00+00:00",
    ]


def test_db_reader_read_latest_no_runs_returns_none():
    reader = DbManifestReader({"database_url": TEST_DATABASE_URL})

    assert reader.read_latest() is None


def test_db_reader_read_recent_missing_table_returns_empty_list():
    reader = DbManifestReader({"database_url": TEST_DATABASE_URL})

    assert reader.read_recent(10) == []


def test_db_reader_read_recent_propagates_real_connection_errors():
    # A missing table (above) is "no runs recorded yet" and returns [], but
    # a genuine connection/auth failure must propagate instead of also
    # silently degrading to an empty history -- otherwise a real
    # operational problem (bad credentials, unreachable host, ...) would
    # look identical to "nothing's run yet" on the dashboard.
    reader = DbManifestReader({"database_url": "postgresql+psycopg://baduser:badpass@127.0.0.1:1/nonexistent"})

    with pytest.raises(OperationalError):
        reader.read_recent(10)


def test_db_reader_nan_columns_from_a_pre_upgrade_row_become_none():
    # Simulates a manifest row written before "status"/"error" existed:
    # to_sql will have left those columns NULL for older rows once the
    # columns are added, which pandas' read_sql surfaces as NaN, not None.
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                'CREATE TABLE "recommendation_runs" ('
                '"generated_at" TEXT, "status" TEXT, "error" TEXT, "n_events" INTEGER'
                ")"
            )
        )
        conn.execute(
            text('INSERT INTO "recommendation_runs" ("generated_at", "n_events") VALUES (:g, :n)'),
            {"g": "2026-07-20T00:00:00+00:00", "n": 5},
        )

    reader = DbManifestReader({"database_url": TEST_DATABASE_URL})

    latest = reader.read_latest()
    assert latest["status"] is None
    assert latest["error"] is None


def test_db_reader_missing_database_url_raises():
    with pytest.raises(RuntimeError, match="database_url"):
        DbManifestReader({})
