from __future__ import annotations

from sqlalchemy import text

from cicerone.io.db_store import DatabaseInputSource
from cicerone.io.engines import dispose_engines, engine_for, release_engine
from cicerone.locks.postgres import PostgresAdvisoryLock
from cicerone.serve.bootstrap_events import EventsRuntime


def test_engine_for_reuses_url(tmp_path):
    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    first = engine_for(url)
    assert engine_for(url) is first
    dispose_engines()
    assert engine_for(url) is not first
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
