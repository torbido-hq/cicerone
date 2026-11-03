from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from support.events import poll_one_by_one
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
    assert source._keyed_source is not None
    assert "ctid" in source._keyed_source
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


def test_postgres_mixed_case_event_columns_poll_and_page():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE TABLE "{_TABLE}" ("User_Id" TEXT, "Item_Id" TEXT, "Event_Type" TEXT, '
                '"Occurred_At" TIMESTAMPTZ, "Event_Id" TEXT)'
            )
        )
        conn.execute(
            text(
                f'INSERT INTO "{_TABLE}" ("User_Id", "Item_Id", "Event_Type", "Occurred_At", "Event_Id") '
                "VALUES "
                "('u1', 'i1', 'purchase', '2026-08-13T12:00:00Z', 'e1'), "
                "('u2', 'i2', 'purchase', '2026-08-13T12:00:00Z', 'e2')"
            )
        )
    source = _source()
    source.connect()

    first = list(source.poll(1))
    source.ack(first)
    second = list(source.poll(1))

    assert [(event.event_id, event.user_id) for event in first + second] == [("e1", "u1"), ("e2", "u2")]


def test_postgres_naive_occurred_at_is_utc_whatever_the_session_time_zone():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE TABLE "{_TABLE}" (user_id TEXT, item_id TEXT, event_type TEXT, '
                "occurred_at TIMESTAMP, event_id TEXT)"
            )
        )
        conn.execute(
            text(
                f'INSERT INTO "{_TABLE}" (user_id, item_id, event_type, occurred_at, event_id) VALUES '
                "('u1', 'i1', 'purchase', '2026-08-13 04:00:00', 'before'), "
                "('u1', 'i2', 'purchase', '2026-08-13 12:00:00', 'after')"
            )
        )
    source = DbEventSource(
        {
            "database_url": f"{TEST_DATABASE_URL}?options=-c%20timezone%3DAsia/Tokyo",
            "events_table": _TABLE,
            "initial_watermark": "2026-08-13T05:00:00Z",
        }
    )
    source.connect()
    try:
        assert source.health().lag == 1
        assert [event.event_id for event in source.poll(10)] == ["after"]
    finally:
        source.close()


def _seed_same_time_rows(extra_columns: str, extras: list[str]) -> None:
    columns = "user_id TEXT, item_id TEXT, event_type TEXT, quantity INTEGER, occurred_at TIMESTAMPTZ"
    rows = ", ".join(
        f"('u1', 'i{n}', 'purchase', 1, '2026-08-13T12:00:00Z'{extra})" for n, extra in enumerate(extras)
    )
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE TABLE "{_TABLE}" ({columns}{extra_columns})'))
        conn.execute(text(f'INSERT INTO "{_TABLE}" VALUES {rows}'))


def test_postgres_same_timestamp_rows_without_event_id_page_past_the_first_page():
    _seed_same_time_rows("", [""] * 12)
    source = _source()
    source.connect()
    seen = poll_one_by_one(source)
    assert len(seen) == len(set(seen)) == 12


def test_postgres_odd_identity_shapes_are_each_delivered_once():
    ids = ["id:-2", "id:-10", "id:1" + "0" * 20, "ID:5", "id:+5", "id:5", "id:10"]
    _seed_same_time_rows(", event_id TEXT", [f", '{event_id}'" for event_id in ids])
    source = _source()
    source.connect()
    assert sorted(poll_one_by_one(source)) == sorted(ids)


def test_postgres_event_id_order_ignores_the_column_collation():
    _seed_same_time_rows(', event_id TEXT COLLATE "und-x-icu"', [f", '{event_id}'" for event_id in "abB"])
    source = _source()
    source.connect()
    assert poll_one_by_one(source) == ["B", "a", "b"]


def test_postgres_microsecond_timestamps_page_in_time_order():
    _seed_same_time_rows(", event_id TEXT", [", 'z'", ", 'a'"])
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        later = "occurred_at + interval '1 microsecond'"
        conn.execute(text(f"UPDATE \"{_TABLE}\" SET occurred_at = {later} WHERE event_id = 'a'"))
    source = _source()
    source.connect()
    assert poll_one_by_one(source) == ["z", "a"]


def test_postgres_keyless_rows_at_one_timestamp_are_each_delivered_as_rejected():
    _seed_same_time_rows(", event_id TEXT", [", NULL"] * 12)
    source = _source()
    source.connect()
    rejected = []
    while batch := list(source.poll(1)):
        rejected.append(batch[0].rejected)
        assert source.ack([batch[0].event_id]) == (batch[0].event_id,)
    assert len(rejected) == 12
    assert all("event_id or idempotency_key is required" in (reason or "") for reason in rejected)


def test_postgres_view_without_ctid_pages_by_id():
    _seed_same_time_rows(", id INTEGER", [f", {n}" for n in range(1, 13)])
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE VIEW "{_TABLE}_view" AS SELECT * FROM "{_TABLE}"'))
    source = DbEventSource({"database_url": TEST_DATABASE_URL, "events_table": f"{_TABLE}_view"})
    source.connect()
    try:
        assert poll_one_by_one(source) == [f"id:{n}" for n in range(1, 13)]
    finally:
        source.close()
        with engine.begin() as conn:
            conn.execute(text(f'DROP VIEW "{_TABLE}_view"'))


def test_postgres_prefilter_keeps_rows_at_the_watermark_time(tmp_path):
    times = {"d": "11:59", "a": "12:00", "b": "12:00", "c": "12:01"}
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE TABLE "{_TABLE}" (user_id TEXT, item_id TEXT, event_type TEXT, '
                "occurred_at TIMESTAMPTZ, event_id TEXT)"
            )
        )
        for event_id, at in times.items():
            conn.execute(
                text(f"INSERT INTO \"{_TABLE}\" VALUES ('u1', :id, 'purchase', :at, :id)"),
                {"id": event_id, "at": f"2026-08-13T{at}:00Z"},
            )
    watermark = tmp_path / "wm.json"
    watermark.write_text('{"occurred_at": "2026-08-13T12:00:00+00:00", "event_id": "a"}\n')
    options = {"database_url": TEST_DATABASE_URL, "events_table": _TABLE, "watermark_path": str(watermark)}
    legacy = DbEventSource(options)
    legacy.connect()
    try:
        assert "occurred_at >= CAST(:watermark_at AS timestamptz)" in legacy._source_sql(engine)
        assert legacy.health().lag == 3
        first = list(legacy.poll(1))
        assert [event.event_id for event in first] == ["a"]
        legacy.ack(["a"])
    finally:
        legacy.close()
    resumed = DbEventSource(options)
    resumed.connect()
    try:
        assert resumed.health().lag == 2
        assert poll_one_by_one(resumed) == ["b", "c"]
    finally:
        resumed.close()
