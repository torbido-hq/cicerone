from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine

from cicerone.events.db import DbEventSource
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
    table_source = DbEventSource({"database_url": url, "events_table": "events"})
    assert table_source._from_clause() == "events"
    query_source = DbEventSource({"database_url": url, "events_query": "SELECT * FROM events"})
    assert query_source._from_clause() == "(SELECT * FROM events) AS cicerone_events_src"


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
    assert source._has_event_id_column is False
    assert source._select_clause is not None
    assert "extra_unused" not in source._select_clause
    assert "occurred_at" in source._select_clause
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
    assert source._has_event_id_column is False
    assert source._select_clause is not None
    assert "rowid" in source._select_clause
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


def test_db_health_lag_none_when_scan_hits_cap(tmp_path, monkeypatch):
    import cicerone.events.db as db_mod

    monkeypatch.setattr(db_mod, "_LAG_SCAN_LIMIT", 2)
    url = _sqlite_url(tmp_path)
    _seed_events(
        url,
        [
            {
                "user_id": "u1",
                "item_id": f"i{i}",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": f"2026-08-13T12:0{i}:00Z",
                "event_id": f"e{i}",
            }
            for i in range(3)
        ],
    )
    source = DbEventSource({"database_url": url, "initial_watermark": "2026-08-01T00:00:00Z"})
    source.connect()
    assert source.health().lag is None


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
        "",
        "   ",
    ],
)
def test_db_rejects_unsafe_events_query(tmp_path, events_query):
    with pytest.raises(ValueError, match="events_query"):
        DbEventSource({"database_url": _sqlite_url(tmp_path), "events_query": events_query})


def test_db_missing_occurred_at_raises_config_error(tmp_path):
    from cicerone.config import ConfigError

    url = _sqlite_url(tmp_path)
    engine = create_engine(url)
    pd.DataFrame([{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1}]).to_sql(
        "events", engine, if_exists="replace", index=False
    )
    source = DbEventSource({"database_url": url})
    source.connect()
    with pytest.raises(ConfigError, match="occurred_at"):
        source.poll(1)
