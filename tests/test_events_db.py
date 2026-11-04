from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import mysql
from support.events import event_payload, poll_one_by_one

from cicerone.config import ConfigError
from cicerone.events.db import DbEventSource
from cicerone.events.db_identity import (
    _row_identity,
    _stable_event_id,
    after_watermark_sql,
    keyed_source_sql,
)
from cicerone.events.normalize import EventNormalizeError
from cicerone.events.registry import build_event_source, registered_event_source_kinds


def _sqlite_url(tmp_path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'events.db'}"


def _seed_events(url: str, rows: list[dict]) -> None:
    engine = create_engine(url)
    pd.DataFrame(rows).to_sql("events", engine, if_exists="replace", index=False)


def test_db_registered_and_build(tmp_path):
    assert "db" in registered_event_source_kinds()
    url = _sqlite_url(tmp_path)
    source = build_event_source("db", {"database_url": url})
    assert isinstance(source, DbEventSource)


def test_db_from_clause_uses_sql_identifier_as_is(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    table_source = DbEventSource({"database_url": url, "events_table": "events"})
    assert table_source._from_clause(engine) == "events"
    assert DbEventSource({"database_url": url, "events_table": "Events"})._from_clause(engine) == '"Events"'
    query_source = DbEventSource({"database_url": url, "events_query": "SELECT * FROM events"})
    assert query_source._from_clause(engine) == "(SELECT * FROM events) AS cicerone_events_src"


def test_db_poll_ack_advances_watermark(tmp_path):
    url = _sqlite_url(tmp_path)
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "e1",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:01:00Z",
                "event_id": "e2",
            },
        ],
    )
    source = DbEventSource({"database_url": url})
    source.connect()
    first = list(source.poll(1))
    assert len(first) == 1
    assert first[0].event_id == "e1"
    assert source.health().lag >= 1
    source.ack([first[0].event_id])
    second = list(source.poll(10))
    assert [event.event_id for event in second] == ["e2"]
    source.ack([second[0].event_id])
    assert list(source.poll(10)) == []


def test_db_same_timestamp_pages_by_event_id(tmp_path):
    url = _sqlite_url(tmp_path)
    ts = "2026-08-13T12:00:00Z"
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": f"i{i}",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": f"e{i}",
            }
            for i in range(1, 6)
        ],
    )
    source = DbEventSource({"database_url": url})
    source.connect()
    seen: list[str] = []
    for _ in range(5):
        batch = list(source.poll(1))
        assert len(batch) == 1
        seen.append(batch[0].event_id)
        source.ack([batch[0].event_id])
    assert seen == ["e1", "e2", "e3", "e4", "e5"]
    assert list(source.poll(10)) == []


def test_db_health_lag_uses_event_id_cursor(tmp_path):
    url = _sqlite_url(tmp_path)
    ts = "2026-08-13T12:00:00Z"
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": "e1",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": "e2",
            },
            {
                "user_id": "u1",
                "item_id": "i3",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": "e3",
            },
        ],
    )
    source = DbEventSource({"database_url": url})
    source.connect()
    assert source.health().lag == 3
    first = list(source.poll(1))
    assert first[0].event_id == "e1"
    source.ack([first[0].event_id])
    assert source.health().lag == 2


def test_db_nack_allows_repoll(tmp_path):
    url = _sqlite_url(tmp_path)
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "e1",
            }
        ],
    )
    source = DbEventSource({"database_url": url})
    source.connect()
    batch = list(source.poll(10))
    assert len(batch) == 1
    source.nack(batch)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]


def test_db_watermark_path_persists(tmp_path):
    url = _sqlite_url(tmp_path)
    watermark = tmp_path / "wm.json"
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "e1",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T13:00:00Z",
                "event_id": "e2",
            },
        ],
    )
    source = DbEventSource({"database_url": url, "watermark_path": str(watermark)})
    source.connect()
    batch = list(source.poll(1))
    source.ack([batch[0].event_id])
    assert watermark.is_file()

    restarted = DbEventSource({"database_url": url, "watermark_path": str(watermark)})
    restarted.connect()
    left = list(restarted.poll(10))
    assert [event.event_id for event in left] == ["e2"]


def test_db_events_query_and_stable_id_without_event_id(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00+00:00",
                "extra_unused": "ignore-me",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": "2026-08-13T13:00:00+00:00",
                "extra_unused": "ignore-me",
            },
        ]
    ).to_sql("raw_events", engine, if_exists="replace", index=False)
    source = DbEventSource(
        {
            "database_url": url,
            "events_query": "SELECT * FROM raw_events",
            "initial_watermark": "2026-08-01T00:00:00Z",
        }
    )
    source.connect()
    assert source.health().lag == 2
    assert source._keyed_source is not None
    assert "extra_unused" not in source._keyed_source
    assert "occurred_at" in source._keyed_source
    polled = list(source.poll(5))
    assert len(polled) == 2
    assert polled[0].occurred_at == datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    assert "|" in polled[0].event_id
    source.ack([polled[0].event_id])
    assert source.health().lag == 1


def test_db_duplicate_payload_without_event_id_uses_rowid(tmp_path):
    url = _sqlite_url(tmp_path)
    row = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00+00:00",
    }
    _seed_events(url, [row, dict(row)])
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    polled = list(source.poll(10))
    assert source._keyed_source is not None
    assert "rowid" in source._keyed_source
    assert len(polled) == 2
    ids = [event.event_id for event in polled]
    assert ids[0] != ids[1]
    assert all(event_id.startswith("rowid:") for event_id in ids)
    source.nack(polled)

    seen: list[str] = []
    for _ in range(2):
        batch = list(source.poll(1))
        assert len(batch) == 1
        seen.append(batch[0].event_id)
        source.ack([batch[0].event_id])
    assert seen[0] != seen[1]
    assert list(source.poll(1)) == []


def test_db_rowid_watermark_survives_restart(tmp_path):
    url = _sqlite_url(tmp_path)
    watermark = tmp_path / "wm.json"
    row = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00+00:00",
    }
    _seed_events(url, [row, dict(row)])
    source = DbEventSource(
        {
            "database_url": url,
            "watermark_path": str(watermark),
            "initial_watermark": "2026-08-01T00:00:00Z",
        }
    )
    source.connect()
    first = list(source.poll(1))
    assert len(first) == 1
    assert first[0].event_id.startswith("rowid:")
    source.ack([first[0].event_id])
    source.close()

    restarted = DbEventSource({"database_url": url, "watermark_path": str(watermark)})
    restarted.connect()
    left = list(restarted.poll(10))
    assert len(left) == 1
    assert left[0].event_id.startswith("rowid:")
    assert left[0].event_id != first[0].event_id


def test_db_duplicate_payload_prefers_id_column_over_rowid(tmp_path):
    url = _sqlite_url(tmp_path)
    row = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00+00:00",
    }
    _seed_events(url, [{**row, "id": 10}, {**row, "id": 20}])
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    polled = list(source.poll(10))
    assert len(polled) == 2
    ids = [event.event_id for event in polled]
    assert ids[0] != ids[1]
    assert set(ids) == {"id:10", "id:20"}


def test_db_events_query_duplicate_payload_collides_without_id(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    row = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00+00:00",
    }
    pd.DataFrame([row, dict(row)]).to_sql("raw_events", engine, if_exists="replace", index=False)
    source = DbEventSource(
        {
            "database_url": url,
            "events_query": "SELECT * FROM raw_events",
            "initial_watermark": "2026-08-01T00:00:00Z",
        }
    )
    source.connect()
    polled = list(source.poll(10))
    assert len(polled) == 1
    assert "|" in polled[0].event_id


def test_db_events_query_duplicate_payload_uses_projected_id(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    row = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00+00:00",
    }
    pd.DataFrame([{**row, "id": 1}, {**row, "id": 2}]).to_sql(
        "raw_events", engine, if_exists="replace", index=False
    )
    source = DbEventSource(
        {
            "database_url": url,
            "events_query": "SELECT * FROM raw_events",
            "initial_watermark": "2026-08-01T00:00:00Z",
        }
    )
    source.connect()
    polled = list(source.poll(10))
    assert len(polled) == 2
    assert {event.event_id for event in polled} == {"id:1", "id:2"}


def test_db_numeric_id_cursor_orders_nine_before_ten(tmp_path):
    url = _sqlite_url(tmp_path)
    row = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00+00:00",
    }
    _seed_events(url, [{**row, "id": 9}, {**row, "id": 10}, {**row, "id": 11}])
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    first = list(source.poll(1))
    assert len(first) == 1
    assert first[0].event_id == "id:9"
    source.ack([first[0].event_id])
    rest = list(source.poll(10))
    assert {event.event_id for event in rest} == {"id:10", "id:11"}


def test_db_event_id_column_numeric_identity_does_not_skip_id_10(tmp_path):
    url = _sqlite_url(tmp_path)
    ts = "2026-08-13T12:00:00+00:00"
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": f"i{n}",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": f"id:{n}",
            }
            for n in (9, 10, 11)
        ],
    )
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    first = list(source.poll(1))
    assert first[0].event_id == "id:9"
    source.ack([first[0].event_id])
    assert source.health().lag == 2
    rest = list(source.poll(10))
    assert {event.event_id for event in rest} == {"id:10", "id:11"}


def test_db_numeric_identity_same_timestamp_fetch_is_bounded(tmp_path):
    url = _sqlite_url(tmp_path)
    ts = "2026-08-13T12:00:00+00:00"
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": f"i{n}",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": f"id:{n}",
            }
            for n in range(1, 41)
        ],
    )
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    first = list(source.poll(1))
    assert first[0].event_id == "id:1"
    source.ack([first[0].event_id])
    for _ in range(8):
        batch = list(source.poll(1))
        source.ack([batch[0].event_id])
    assert source._watermark_event_id == "id:9"
    assert source._engine is not None
    rows = source._fetch_rows(source._engine, source._watermark_cursor, source._watermark_at, limit=5)
    assert len(rows) == 5
    assert [row["event_id"] for row in rows] == [f"id:{n}" for n in range(10, 15)]


def test_db_subsecond_timestamps_do_not_skip_later_event(tmp_path):
    url = _sqlite_url(tmp_path)
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00.100000+00:00",
                "event_id": "id:2",
            },
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00.200000+00:00",
                "event_id": "id:1",
            },
        ],
    )
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    first = list(source.poll(1))
    assert first[0].event_id == "id:2"
    source.ack([first[0].event_id])
    rest = list(source.poll(10))
    assert [event.event_id for event in rest] == ["id:1"]


def test_db_prefixed_non_numeric_event_id_does_not_skip_later_row(tmp_path):
    url = _sqlite_url(tmp_path)
    ts = "2026-08-13T12:00:00+00:00"
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "ia",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": "id:order-123",
            },
            {
                "user_id": "u1",
                "item_id": "ib",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": ts,
                "event_id": "id:zzz",
            },
        ],
    )
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    first = list(source.poll(1))
    assert first[0].event_id == "id:order-123"
    source.ack([first[0].event_id])
    rest = list(source.poll(10))
    assert [event.event_id for event in rest] == ["id:zzz"]


def test_row_identity_prefers_id_then_rowid_then_ctid():
    occurred = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    assert _row_identity({"id": 7, "rowid": 1, "ctid": "(0,1)"}) == "id:7"
    assert _row_identity({"rowid": 2}) == "rowid:00000000000000000002"
    assert _row_identity({"rowid": "not-an-int"}) == "rowid:not-an-int"
    assert _row_identity({"rowid": float("inf")}) == "rowid:inf"
    assert _row_identity({"ctid": "(0,1)"}) == "ctid:(0,1)"
    assert _stable_event_id({"idempotency_key": "ik", "id": 1}, occurred) == "ik"
    assert _stable_event_id({"id": 9}, occurred) == "id:9"


def test_db_corrupt_watermark_falls_back_to_initial(tmp_path, caplog):
    import logging

    url = _sqlite_url(tmp_path)
    watermark = tmp_path / "wm.json"
    watermark.write_text("not-json\n")
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "e1",
            }
        ],
    )
    source = DbEventSource(
        {
            "database_url": url,
            "watermark_path": str(watermark),
            "initial_watermark": "2026-08-01T00:00:00Z",
        }
    )
    with caplog.at_level(logging.ERROR):
        source.connect()
    assert any("corrupt watermark" in record.getMessage().lower() for record in caplog.records)
    assert [event.event_id for event in source.poll(10)] == ["e1"]


def test_db_requires_database_url():
    with pytest.raises(ValueError, match="database_url"):
        DbEventSource({})


def test_db_poll_before_connect_raises(tmp_path):
    source = DbEventSource({"database_url": _sqlite_url(tmp_path)})
    with pytest.raises(RuntimeError, match="connect"):
        source.poll(1)


def test_db_close_disposes_engine(tmp_path):
    url = _sqlite_url(tmp_path)
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "e1",
            }
        ],
    )
    source = DbEventSource({"database_url": url})
    source.connect()
    assert list(source.poll(1))
    source.close()
    with pytest.raises(RuntimeError, match="connect"):
        source.poll(1)


def test_db_rejects_timezone_less_initial_watermark(tmp_path):
    with pytest.raises(EventNormalizeError, match="timezone"):
        DbEventSource(
            {
                "database_url": _sqlite_url(tmp_path),
                "initial_watermark": "2026-08-01T00:00:00",
            }
        )


@pytest.mark.parametrize(
    "events_query",
    [
        "DELETE FROM events",
        "SELECT 1; DROP TABLE events",
        "INSERT INTO events SELECT * FROM events",
        "SELECT * FROM events INTO dump",
        "SELECT pg_read_file('/etc/passwd')",
        "",
        "   ",
    ],
)
def test_db_rejects_unsafe_events_query(tmp_path, events_query):
    with pytest.raises(ValueError, match="events_query"):
        DbEventSource({"database_url": _sqlite_url(tmp_path), "events_query": events_query})


def test_db_missing_occurred_at_raises_config_error(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1}]).to_sql(
        "events", engine, if_exists="replace", index=False
    )
    source = DbEventSource({"database_url": url})
    source.connect()
    with pytest.raises(ConfigError, match="occurred_at"):
        source.poll(1)


def test_db_health_sql_error_keeps_local_lag(tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError

    url = _sqlite_url(tmp_path)
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "e1",
            }
        ],
    )
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    assert list(source.poll(1))

    def boom(engine, watermark_at, watermark_event_id):
        raise OperationalError("count", {}, Exception("down"))

    monkeypatch.setattr(source, "_count_after", boom)
    assert source.health().lag == 1


def test_db_health_unexpected_error_propagates(tmp_path, monkeypatch):
    url = _sqlite_url(tmp_path)
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-13T12:00:00Z",
                "event_id": "e1",
            }
        ],
    )
    source = DbEventSource({"database_url": url})
    source.connect()

    def boom(engine, watermark_at, watermark_event_id):
        raise RuntimeError("lag bug")

    monkeypatch.setattr(source, "_count_after", boom)
    with pytest.raises(RuntimeError, match="lag bug"):
        source.health()


def test_db_unexpected_watermark_error_propagates(tmp_path, monkeypatch):
    from cicerone.io.engines import release_engine

    watermark = tmp_path / "wm.json"
    watermark.write_text('{"occurred_at": "2026-08-01T00:00:00Z", "event_id": "e0"}\n')
    url = _sqlite_url(tmp_path)
    source = DbEventSource({"database_url": url, "watermark_path": str(watermark)})
    released: list[str] = []
    real_release = release_engine

    def spy(database_url: str) -> None:
        released.append(database_url)
        real_release(database_url)

    monkeypatch.setattr("cicerone.events.db.release_engine", spy)
    real = type(watermark).read_text

    def boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == watermark:
            raise RuntimeError("watermark bug")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(watermark), "read_text", boom)
    with pytest.raises(RuntimeError, match="watermark bug"):
        source.connect()
    assert source._engine is None
    assert source._connected is False
    assert released == [url]
    source.close()
    assert released == [url]


def test_db_connect_keeps_existing_engine_when_watermark_fails(tmp_path, monkeypatch):
    from cicerone.io.engines import release_engine

    url = _sqlite_url(tmp_path)
    source = DbEventSource({"database_url": url})
    source.connect()
    engine = source._engine
    released: list[str] = []
    real_release = release_engine

    def spy(database_url: str) -> None:
        released.append(database_url)
        real_release(database_url)

    monkeypatch.setattr("cicerone.events.db.release_engine", spy)

    def boom() -> None:
        raise RuntimeError("watermark bug")

    monkeypatch.setattr(source, "_load_watermark_unlocked", boom)
    with pytest.raises(RuntimeError, match="watermark bug"):
        source.connect()
    assert source._engine is engine
    assert released == []
    source.close()
    assert released == [url]


def _event_row(name: str, minute: int, **overrides) -> dict:
    """Row ``name`` (its event_id unless ``event_id=`` overrides it) at 12:``minute``."""
    defaults = {
        "item_id": f"i{name}",
        "occurred_at": f"2026-08-13T12:{minute:02d}:00Z",
        "event_id": name,
    }
    return event_payload(**{**defaults, **overrides})


def test_db_poll_delivers_an_invalid_row_as_rejected_in_cursor_order(tmp_path):
    url = _sqlite_url(tmp_path)
    bad = _event_row("bad", 1, user_id="alice@example.com", quantity=0)
    _seed_events(url, [_event_row("e1", 0), bad, _event_row("e2", 2)])
    source = DbEventSource({"database_url": url})
    source.connect()
    first = list(source.poll(2))
    assert [(event.event_id, event.rejected is None) for event in first] == [("e1", True), ("bad", False)]
    reason = first[1].rejected or ""
    assert "quantity must be >= 1" in reason
    assert '"event_id": "bad"' in reason
    assert "alice@example.com" not in reason
    assert "ibad" not in reason
    assert source.ack([event.event_id for event in first]) == ("e1", "bad")
    assert [event.event_id for event in source.poll(10)] == ["e2"]


def test_db_poll_keeps_an_unacked_rejected_row_in_flight(tmp_path):
    url = _sqlite_url(tmp_path)
    _seed_events(url, [_event_row("bad", 0, quantity=0)])
    source = DbEventSource({"database_url": url})
    source.connect()
    assert [event.event_id for event in source.poll(10)] == ["bad"]
    assert list(source.poll(10)) == []
    source.ack(["bad"])
    assert list(source.poll(10)) == []
    assert source.health().lag == 0


def test_db_rows_with_a_garbage_occurred_at_never_stall_the_source(tmp_path):
    url = _sqlite_url(tmp_path)
    junk = [_event_row(f"junk{n}", 0, occurred_at="not a time") for n in range(5)]
    _seed_events(url, [*junk, _event_row("e1", 1)])
    source = DbEventSource({"database_url": url})
    source.connect()

    assert [event.event_id for event in source.poll(2)] == ["e1"]
    assert source.health().lag == 1


def _watermarked_source(tmp_path, rows: list[dict]) -> tuple[DbEventSource, dict]:
    url = _sqlite_url(tmp_path)
    _seed_events(url, rows)
    options = {"database_url": url, "watermark_path": str(tmp_path / "wm.json")}
    source = DbEventSource(options)
    source.connect()
    return source, options


def _restarted_ids(options: dict) -> list[str]:
    restarted = DbEventSource(options)
    restarted.connect()
    return [event.event_id for event in restarted.poll(10)]


def _saved_event_id(tmp_path) -> str | None:
    watermark = tmp_path / "wm.json"
    return json.loads(watermark.read_text())["event_id"] if watermark.exists() else None


def test_db_out_of_order_ack_does_not_lose_earlier_events_on_restart(tmp_path):
    source, options = _watermarked_source(tmp_path, [_event_row(f"e{n}", n) for n in (1, 2, 3)])
    assert [event.event_id for event in source.poll(10)] == ["e1", "e2", "e3"]
    assert source.ack(["e3"]) == ("e3",)
    assert _saved_event_id(tmp_path) is None
    assert _restarted_ids(options) == ["e1", "e2", "e3"]
    assert source.ack(["e1"]) == ("e1",)
    assert _saved_event_id(tmp_path) == "e1"
    assert source.ack(["e2", "e3"]) == ("e2",)
    assert _saved_event_id(tmp_path) == "e3"
    assert _restarted_ids(options) == []


def test_db_nacked_head_is_redelivered_and_holds_the_watermark(tmp_path):
    source, options = _watermarked_source(tmp_path, [_event_row(f"e{n}", n) for n in (1, 2, 3)])
    first, *_rest = source.poll(10)
    assert source.nack([first]) == ()
    assert source.ack(["e1", "e2", "e3"]) == ("e2", "e3")
    assert _saved_event_id(tmp_path) is None
    assert [event.event_id for event in source.poll(10)] == ["e1"]
    assert source.ack(["e1"]) == ("e1",)
    assert _saved_event_id(tmp_path) == "e3"
    assert _restarted_ids(options) == []


def test_db_vanished_head_row_is_dropped_so_the_watermark_moves(tmp_path, caplog):
    import logging

    source, options = _watermarked_source(tmp_path, [_event_row(f"e{n}", n) for n in (1, 2, 3)])
    first, *_rest = source.poll(10)
    source.nack([first])
    source.ack(["e2"])
    _seed_events(options["database_url"], [_event_row(f"e{n}", n) for n in (2, 3)])
    with caplog.at_level(logging.WARNING, logger="cicerone.events.db"):
        assert list(source.poll(10)) == []
    assert any("e1" in record.getMessage() for record in caplog.records)
    assert _saved_event_id(tmp_path) == "e2"
    assert source.ack(["e3"]) == ("e3",)
    assert _restarted_ids(options) == []


def test_db_head_beyond_a_truncated_page_is_not_dropped(tmp_path):
    source, options = _watermarked_source(tmp_path, [_event_row("e9", 9)])
    assert [event.event_id for event in source.poll(10)] == ["e9"]
    late = [_event_row(f"late{n:02d}", 1, item_id=f"l{n}") for n in range(20)]
    _seed_events(options["database_url"], [*late, _event_row("e9", 9)])
    assert len(source.poll(1)) == 1
    assert source.ack(["e9"]) == ("e9",)


def test_db_same_timestamp_watermark_follows_event_id_order(tmp_path):
    source, options = _watermarked_source(tmp_path, [_event_row(f"e{n}", 0) for n in (3, 2)])
    assert [event.event_id for event in source.poll(10)] == ["e2", "e3"]
    _seed_events(options["database_url"], [_event_row(f"e{n}", 0) for n in (3, 1, 2)])
    assert [event.event_id for event in source.poll(10)] == ["e1"]
    assert source.ack(["e3", "e2"]) == ("e3", "e2")
    assert _saved_event_id(tmp_path) is None
    assert _restarted_ids(options) == ["e1", "e2", "e3"]
    source.ack(["e1"])
    assert _saved_event_id(tmp_path) == "e3"


def test_db_health_reports_acked_events_stuck_behind_the_head(tmp_path, monkeypatch):
    import cicerone.events.db as db_mod

    monkeypatch.setattr(db_mod, "monotonic", lambda: 100.0)
    source, _options = _watermarked_source(tmp_path, [_event_row(f"e{n}", n) for n in (1, 2, 3)])
    source.poll(10)
    source.ack(["e2", "e3"])
    monkeypatch.setattr(db_mod, "monotonic", lambda: 130.0)
    assert "acked_behind_head=2 head_wait=30s" in (source.health().detail or "")
    source.ack(["e1"])
    assert "acked_behind_head" not in (source.health().detail or "")


def test_db_idempotency_key_rows_at_the_watermark_timestamp_are_not_skipped(tmp_path):
    rows = [
        _event_row("e1", 0, idempotency_key=None),
        _event_row("k2", 0, event_id=None, idempotency_key="k2"),
        _event_row("k3", 0, event_id="", idempotency_key="k3"),
        _event_row("e4", 0, idempotency_key=None),
        _event_row("e2", 0, event_id=" e2\t", idempotency_key=None),
        _event_row("k5", 0, event_id="  ", idempotency_key=" k5 "),
    ]
    source, _options = _watermarked_source(tmp_path, rows)
    assert poll_one_by_one(source) == ["e1", "e2", "e4", "k2", "k3", "k5"]


def test_db_zero_event_id_falls_back_to_idempotency_key_in_sql_too(tmp_path):
    rows = [
        _event_row("k0", 0, event_id=0, idempotency_key="k0"),
        _event_row("k5", 0, event_id=5, idempotency_key="k5"),
    ]
    source, _options = _watermarked_source(tmp_path, rows)
    assert poll_one_by_one(source) == ["5", "k0"]


def test_db_row_without_event_id_or_idempotency_key_is_rejected_without_blocking(tmp_path):
    rows = [_event_row("e1", 0), _event_row("bad", 0, event_id=None), _event_row("e3", 1)]
    source, options = _watermarked_source(tmp_path, rows)
    first = list(source.poll(2))
    assert [event.event_id for event in first] == ["e1", ""]
    assert "event_id or idempotency_key is required" in (first[1].rejected or "")
    assert source.ack(["e1", ""]) == ("e1", "")
    assert poll_one_by_one(source) == ["e3"]
    assert _saved_event_id(tmp_path) == "e3"
    assert _restarted_ids(options) == []


def test_db_same_timestamp_rows_without_event_id_page_past_the_first_page(tmp_path):
    row = {"user_id": "u1", "event_type": "purchase", "quantity": 1, "occurred_at": "2026-08-13T12:00:00Z"}
    url = _sqlite_url(tmp_path)
    _seed_events(url, [{**row, "item_id": f"i{n}"} for n in range(12)])
    source = DbEventSource({"database_url": url})
    source.connect()
    seen = poll_one_by_one(source)
    assert len(seen) == len(set(seen)) == 12


def test_db_odd_identity_shapes_are_each_delivered_once(tmp_path):
    ids = ["id:-2", "id:-10", "id:1" + "0" * 20, "ID:5", "id:+5", "id:5", "id:10"]
    source, _options = _watermarked_source(tmp_path, [_event_row(event_id, 0) for event_id in ids])
    seen = poll_one_by_one(source)
    assert sorted(seen) == sorted(ids)


def test_db_sub_millisecond_timestamps_page_in_time_order(tmp_path):
    rows = [
        _event_row("z", 0, occurred_at="2026-08-13T12:00:00.100100+00:00"),
        _event_row("a", 0, occurred_at="2026-08-13T14:00:00.100400+02:00"),
        _event_row("m", 0, occurred_at="2026-08-13T12:00:00.1002Z"),
    ]
    source, _options = _watermarked_source(tmp_path, rows)
    assert poll_one_by_one(source) == ["z", "m", "a"]


def test_db_padded_event_ids_page_by_their_trimmed_value(tmp_path):
    source, _options = _watermarked_source(tmp_path, [_event_row(" x", 0), _event_row("!x", 0)])
    assert poll_one_by_one(source) == ["!x", "x"]


def test_db_ctid_shaped_event_ids_are_each_delivered_once(tmp_path):
    ids = ["CTID:(0,1)", "ctid:(0,2)", "ctid:(0,10)", "ctid:a b", "ctid:ab", "ctid:", "ctid:(0,x)"]
    source, _options = _watermarked_source(tmp_path, [_event_row(event_id, 0) for event_id in ids])
    seen = poll_one_by_one(source)
    assert sorted(seen) == sorted(ids)


def test_db_keyless_rows_at_one_timestamp_are_each_delivered_as_rejected(tmp_path):
    keyless = [_event_row(f"k{n}", 0, event_id=None, item_id=f"i{n}") for n in range(12)]
    source, options = _watermarked_source(tmp_path, [*keyless, _event_row("late", 1)])
    rejected = []
    while batch := list(source.poll(1)):
        rejected.append(batch[0].rejected)
        assert source.ack([batch[0].event_id]) == (batch[0].event_id,)
    assert len(rejected) == 13
    assert all("event_id or idempotency_key is required" in (reason or "") for reason in rejected[:12])
    assert rejected[12] is None
    assert _saved_event_id(tmp_path) == "late"
    assert _restarted_ids(options) == []


@pytest.mark.parametrize("database_url", ["mssql+pyodbc://user@localhost/events", "oracle://user@localhost"])
def test_db_rejects_dialects_without_a_sql_cursor(database_url):
    with pytest.raises(ConfigError, match="sqlite, postgresql and mysql/mariadb"):
        DbEventSource({"database_url": database_url})


@pytest.mark.parametrize(
    "database_url", ["mysql+pymysql://user@localhost/events", "mariadb://user@localhost/e"]
)
def test_db_mysql_cursor_sql_concatenates_formats_in_utc_and_compares_binary(database_url):
    DbEventSource({"database_url": database_url})
    columns = {name: f"`{name}`" for name in ("user_id", "occurred_at", "event_id", "id")}
    where, order = after_watermark_sql("mysql")
    sql = text(
        f"SELECT * FROM {keyed_source_sql('mysql', '`events`', columns)} WHERE {where} ORDER BY {order}"
    )
    compiled = str(sql.compile(dialect=mysql.dialect()))
    assert "||" not in compiled
    assert "AS TEXT" not in compiled
    assert compiled.count("DATE_FORMAT(occurred_at, '%%Y-%%m-%%d %%H:%%i:%%s.%%f')") == 1
    assert "CONCAT('id:', NULLIF(CAST(`id` AS CHAR), ''))" in compiled
    assert "CONCAT('id:', LPAD(SUBSTRING(cicerone_identity, 4), 20, '0'), CHAR(31 USING utf8mb4)" in compiled
    assert "FROM `events` WHERE `occurred_at` >= %s)" in compiled
    assert "WHERE cicerone_cursor COLLATE utf8mb4_bin > COALESCE(%s, DATE_FORMAT(%s," in compiled
    assert compiled.endswith("ORDER BY cicerone_cursor COLLATE utf8mb4_bin ASC")


@pytest.mark.parametrize(
    ("extra", "events_query"),
    [({"Cicerone_Cursor": "x"}, None), ({}, "SELECT *, 'x' AS cicerone_cursor FROM events")],
)
def test_db_source_column_named_like_the_cursor_is_a_config_error(tmp_path, extra, events_query):
    url = _sqlite_url(tmp_path)
    _seed_events(url, [{**_event_row("e1", 0), **extra}])
    source = DbEventSource({"database_url": url, "events_query": events_query})
    source.connect()
    with pytest.raises(ConfigError, match="cicerone_cursor"):
        source.poll(1)


def test_db_without_rowid_table_pages_by_id(tmp_path):
    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, user_id TEXT, item_id TEXT,"
                " event_type TEXT, quantity INTEGER, occurred_at TEXT) WITHOUT ROWID"
            )
        )
        for n in range(1, 13):
            conn.execute(
                text("INSERT INTO events VALUES (:id, 'u1', :item, 'purchase', 1, '2026-08-13T12:00:00Z')"),
                {"id": n, "item": f"i{n}"},
            )
    source = DbEventSource({"database_url": url})
    source.connect()
    assert poll_one_by_one(source) == [f"id:{n}" for n in range(1, 13)]


def test_db_events_query_projecting_id_pages_past_the_first_page(tmp_path):
    url = _sqlite_url(tmp_path)
    _seed_events(url, [_event_row(f"e{n}", 0, id=n) for n in range(1, 13)])
    query = "SELECT id, user_id, item_id, event_type, quantity, occurred_at FROM events"
    source = DbEventSource({"database_url": url, "events_query": query})
    source.connect()
    assert poll_one_by_one(source) == [f"id:{n}" for n in range(1, 13)]


def test_db_watermark_file_stores_the_sql_cursor_and_resumes_after_it(tmp_path):
    rows = [_event_row(f"e{n}", 0) for n in (1, 2, 3)]
    source, options = _watermarked_source(tmp_path, rows)
    assert [event.event_id for event in source.poll(2)] == ["e1", "e2"]
    source.ack(["e1", "e2"])
    saved = json.loads((tmp_path / "wm.json").read_text())
    assert saved["event_id"] == "e2"
    assert saved["cursor"].endswith("e2")
    assert _restarted_ids(options) == ["e3"]


def test_db_watermark_file_without_cursor_resumes_at_its_timestamp(tmp_path):
    rows = [_event_row("e0", 0), _event_row("e1", 1), _event_row("e2", 1), _event_row("e3", 2)]
    _source, options = _watermarked_source(tmp_path, rows)
    (tmp_path / "wm.json").write_text('{"occurred_at": "2026-08-13T12:01:00+00:00", "event_id": "e2"}\n')
    assert _restarted_ids(options) == ["e1", "e2", "e3"]
