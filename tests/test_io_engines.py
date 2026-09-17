from __future__ import annotations

from cicerone.io.db_store import DatabaseInputSource
from cicerone.io.engines import dispose_engines, engine_for
from cicerone.locks.postgres import PostgresAdvisoryLock


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
