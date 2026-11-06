from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from cicerone.errors import log_caught
from cicerone.io import engines as io_engines
from cicerone.io.db_store import DatabaseInputSource
from cicerone.io.engines import dispose_engines, engine_for, quote_identifier, release_engine
from cicerone.job_output import truncate_job_error
from cicerone.locks.postgres import PostgresAdvisoryLock
from cicerone.serve.bootstrap_events import EventsRuntime


def test_engine_for_reuses_url(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    first = engine_for(url)
    assert engine_for(url) is first
    dispose_engines()
    assert engine_for(url) is not first
    dispose_engines()


def test_engine_for_hides_bound_values_in_errors(tmp_path, caplog):
    engine = engine_for(f"sqlite+pysqlite:///{tmp_path / 'shared.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (user_id TEXT UNIQUE, item_id TEXT)"))
    insert = text("INSERT INTO t (user_id, item_id) VALUES (:user_id, :item_id)")
    row = {"user_id": "alice@example.com", "item_id": "secret-item"}
    with engine.begin() as conn:
        conn.execute(insert, row)

    with pytest.raises(IntegrityError) as caught, engine.begin() as conn:
        conn.execute(insert, row)

    log_caught("Store unavailable", caught.value)
    for surfaced in (str(caught.value), truncate_job_error(caught.value), caplog.text):
        assert "alice@example.com" not in surfaced
        assert "secret-item" not in surfaced
    assert "INSERT INTO t (user_id, item_id)" in str(caught.value)
    dispose_engines()


def test_readers_and_lock_share_engine(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    dispose_engines()
    source = DatabaseInputSource({"database_url": url})
    lock = PostgresAdvisoryLock(url)
    assert source._engine is lock._engine
    dispose_engines()


def test_close_does_not_dispose_shared_engine(tmp_path):
    from cicerone.events.db import DbEventSource

    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    dispose_engines()
    first = engine_for(url)
    source = DbEventSource({"database_url": url})
    source.connect()
    source.close()
    assert engine_for(url) is first
    with first.connect() as conn:
        conn.execute(text("SELECT 1"))
    dispose_engines()


def test_release_engine_disposes_last_checkout(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    dispose_engines()
    first = engine_for(url)
    release_engine(url)
    assert engine_for(url) is not first
    dispose_engines()


def test_release_engine_keeps_other_checkouts(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    dispose_engines()
    first = engine_for(url)
    engine_for(url)
    release_engine(url)
    assert engine_for(url) is first
    dispose_engines()


def test_events_runtime_stop_keeps_shared_engine(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    dispose_engines()
    first = engine_for(url)
    assert EventsRuntime(webhook_source=None, worker=None).stop() is True
    assert engine_for(url) is first
    dispose_engines()


def test_engine_cache_keeps_all_urls(tmp_path):
    engines = []
    for i in range(10):
        path = tmp_path / f"e{i}.db"
        sqlite3.connect(path).close()
        engines.append(engine_for(f"sqlite+pysqlite:///{path}"))
    assert len(io_engines._engines) == 10
    assert engine_for(f"sqlite+pysqlite:///{tmp_path / 'e0.db'}") is engines[0]


def test_quote_identifier_follows_the_dialect():
    from types import SimpleNamespace

    from sqlalchemy.dialects import mysql, postgresql

    postgres = SimpleNamespace(dialect=postgresql.dialect())
    assert quote_identifier(postgres, "recommendations") == "recommendations"
    assert quote_identifier(postgres, "Mixed_Case") == '"Mixed_Case"'
    assert quote_identifier(postgres, "user") == '"user"'
    assert quote_identifier(SimpleNamespace(dialect=mysql.dialect()), "Mixed_Case") == "`Mixed_Case`"
