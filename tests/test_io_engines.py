from __future__ import annotations

import pytest
from sqlalchemy import text

from cicerone.io.db_store import DatabaseInputSource
from cicerone.io.engines import dispose_engines, engine_for, quote_identifier, release_engine
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


def test_quote_identifier_follows_the_dialect():
    from types import SimpleNamespace

    from sqlalchemy.dialects import mysql, postgresql

    postgres = SimpleNamespace(dialect=postgresql.dialect())
    assert quote_identifier(postgres, "recommendations") == "recommendations"
    assert quote_identifier(postgres, "Mixed_Case") == '"Mixed_Case"'
    assert quote_identifier(postgres, "user") == '"user"'
    assert quote_identifier(SimpleNamespace(dialect=mysql.dialect()), "Mixed_Case") == "`Mixed_Case`"


@pytest.mark.parametrize(
    ("dialect", "clause"),
    [
        ("postgresql", "ON CONFLICT DO NOTHING"),
        ("sqlite", "ON CONFLICT DO NOTHING"),
        ("mysql", "ON DUPLICATE KEY UPDATE event_id = VALUES(event_id)"),
    ],
)
def test_insert_ignoring_duplicates_uses_the_dialect_syntax(dialect, clause):
    from types import SimpleNamespace

    from sqlalchemy.dialects import mysql, postgresql, sqlite

    from cicerone.io.engines import insert_ignoring_duplicates
    from cicerone.track.store_db import track_table

    module = {"postgresql": postgresql, "sqlite": sqlite, "mysql": mysql}[dialect]
    bind = SimpleNamespace(dialect=module.dialect())

    sql = str(insert_ignoring_duplicates(bind, track_table("track")).compile(dialect=module.dialect()))

    assert clause in sql
