"""Shared SQLAlchemy engines keyed by database URL."""

from __future__ import annotations

import threading
from collections.abc import Callable

import sqlalchemy
from sqlalchemy import Connection, Engine, Insert, String, Table, Text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

_engines: dict[str, Engine] = {}
_checkouts: dict[str, int] = {}
_engines_lock = threading.Lock()


def engine_for(database_url: str) -> Engine:
    with _engines_lock:
        engine = _engines.get(database_url)
        if engine is None:
            engine = sqlalchemy.create_engine(database_url, pool_pre_ping=True)
            _engines[database_url] = engine
            _checkouts[database_url] = 0
        _checkouts[database_url] += 1
        return engine


def quote_identifier(bind: Engine | Connection, name: str) -> str:
    """``name`` quoted the way the dialect (and pandas/SQLAlchemy table creation) quotes it."""
    return bind.dialect.identifier_preparer.quote(name)


# MySQL can't index TEXT without a length; binary collation keeps keys case-sensitive as elsewhere.
TEXT_KEY = Text().with_variant(String(255, collation="utf8mb4_bin"), "mysql")


def insert_ignoring_duplicates(bind: Engine | Connection, table: Table) -> Insert:
    """INSERT that skips rows whose primary key already exists, in the dialect's own syntax."""
    dialect = bind.dialect.name
    if dialect == "postgresql":
        return postgresql_insert(table).on_conflict_do_nothing()
    if dialect == "sqlite":
        return sqlite_insert(table).on_conflict_do_nothing()
    # Not INSERT IGNORE: that also turns truncation and other errors into warnings.
    statement = mysql_insert(table)
    key = next(iter(table.primary_key)).name
    return statement.on_duplicate_key_update({key: statement.inserted[key]})


def sql_source(bind: Engine | Connection, query: str | None, table: str, alias: str) -> str:
    """A configured read-only SELECT as an aliased subquery, else the quoted table."""
    if query:
        return f"({query.strip().rstrip(';').strip()}) AS {alias}"
    return quote_identifier(bind, table)


def release_engine(database_url: str) -> None:
    with _engines_lock:
        remaining = _checkouts.get(database_url)
        if remaining is None:
            return
        remaining -= 1
        if remaining > 0:
            _checkouts[database_url] = remaining
            return
        engine = _engines.pop(database_url, None)
        _checkouts.pop(database_url, None)
    if engine is not None:
        engine.dispose()


def dispose_engines() -> None:
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
        _checkouts.clear()
    for engine in engines:
        engine.dispose()


class LazyEngine:
    """Check out ``engine_for(url)`` once, on first use, even under concurrent first calls."""

    def __init__(self, database_url: Callable[[], str]):
        self._database_url = database_url
        self._engine: Engine | None = None
        self._lock = threading.Lock()

    def get(self) -> Engine:
        engine = self._engine
        if engine is not None:
            return engine
        with self._lock:
            if self._engine is None:
                self._engine = engine_for(self._database_url())
            return self._engine
