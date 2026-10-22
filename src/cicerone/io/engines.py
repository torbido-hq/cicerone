"""Shared SQLAlchemy engines keyed by database URL."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import sqlalchemy
from sqlalchemy import Connection, Engine

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


def create_table_sql(bind: Engine | Connection, table: str, columns: Sequence[tuple[str, str]]) -> str:
    """``CREATE TABLE IF NOT EXISTS`` for ``(name, type)`` columns, every identifier quoted."""
    spec = ", ".join(f"{quote_identifier(bind, name)} {kind}" for name, kind in columns)
    return f"CREATE TABLE IF NOT EXISTS {quote_identifier(bind, table)} ({spec})"


def insert_sql(bind: Engine | Connection, table: str, names: Sequence[str]) -> str:
    """``INSERT`` of one row bound by column name (``:name``), every identifier quoted."""
    columns = ", ".join(quote_identifier(bind, name) for name in names)
    values = ", ".join(f":{name}" for name in names)
    return f"INSERT INTO {quote_identifier(bind, table)} ({columns}) VALUES ({values})"


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
