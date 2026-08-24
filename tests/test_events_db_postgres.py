from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from support.postgres_defaults import resolve_test_database_url

from cicerone.events.db import DbEventSource

TEST_DATABASE_URL = resolve_test_database_url()

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — DB-backed tests run against "
    "a real Postgres in CI (see docker-compose.ci.yml). Set POSTGRES_TEST_HOST=localhost "
    "locally (see CONTRIBUTING.md).",
)

_TABLE = "events_identity"


@pytest.fixture(autouse=True)
def _clean_table():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{_TABLE}"'))
    yield
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{_TABLE}"'))
    engine.dispose()


def _source() -> DbEventSource:
    return DbEventSource(
        {
            "database_url": TEST_DATABASE_URL,
            "events_table": _TABLE,
            "initial_watermark": "2026-08-01T00:00:00Z",
        }
    )


def test_postgres_duplicate_payload_without_event_id_uses_ctid():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE "{_TABLE}" (
                    user_id TEXT,
                    item_id TEXT,
                    event_type TEXT,
                    quantity INTEGER,
                    occurred_at TIMESTAMPTZ
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                INSERT INTO "{_TABLE}" (user_id, item_id, event_type, quantity, occurred_at)
                VALUES
                    ('u1', 'i1', 'purchase', 1, '2026-08-13T12:00:00Z'),
                    ('u1', 'i1', 'purchase', 1, '2026-08-13T12:00:00Z')
                """
            )
        )
    source = _source()
    source.connect()
    polled = list(source.poll(10))
    assert source._has_event_id_column is False
    assert source._select_clause is not None
    assert "ctid" in source._select_clause
    assert len(polled) == 2
    ids = [event.event_id for event in polled]
    assert ids[0] != ids[1]
    assert all(event_id.startswith("ctid:") for event_id in ids)


def test_postgres_duplicate_payload_prefers_id_column_over_ctid():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE "{_TABLE}" (
                    id INTEGER,
                    user_id TEXT,
                    item_id TEXT,
                    event_type TEXT,
                    quantity INTEGER,
                    occurred_at TIMESTAMPTZ
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                INSERT INTO "{_TABLE}" (id, user_id, item_id, event_type, quantity, occurred_at)
                VALUES
                    (10, 'u1', 'i1', 'purchase', 1, '2026-08-13T12:00:00Z'),
                    (20, 'u1', 'i1', 'purchase', 1, '2026-08-13T12:00:00Z')
                """
            )
        )
    source = _source()
    source.connect()
    polled = list(source.poll(10))
    assert len(polled) == 2
    assert {event.event_id for event in polled} == {"id:10", "id:20"}


def test_postgres_event_id_column_numeric_identity_does_not_skip_id_10():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE "{_TABLE}" (
                    user_id TEXT,
                    item_id TEXT,
                    event_type TEXT,
                    quantity INTEGER,
                    occurred_at TIMESTAMPTZ,
                    event_id TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                INSERT INTO "{_TABLE}"
                    (user_id, item_id, event_type, quantity, occurred_at, event_id)
                VALUES
                    ('u1', 'i9', 'purchase', 1, '2026-08-13T12:00:00Z', 'id:9'),
                    ('u1', 'i10', 'purchase', 1, '2026-08-13T12:00:00Z', 'id:10'),
                    ('u1', 'i11', 'purchase', 1, '2026-08-13T12:00:00Z', 'id:11')
                """
            )
        )
    source = _source()
    source.connect()
    first = list(source.poll(1))
    assert first[0].event_id == "id:9"
    source.ack([first[0].event_id])
    assert source.health().lag == 2
    rest = list(source.poll(10))
    assert {event.event_id for event in rest} == {"id:10", "id:11"}
