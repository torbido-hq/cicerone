"""Shared SQLAlchemy engines keyed by database URL."""

from __future__ import annotations

import threading

import sqlalchemy
from sqlalchemy import Engine

_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def engine_for(database_url: str) -> Engine:
    with _engines_lock:
        engine = _engines.get(database_url)
        if engine is not None:
            return engine
        engine = sqlalchemy.create_engine(database_url, pool_pre_ping=True)
        _engines[database_url] = engine
        return engine


def dispose_engines() -> None:
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
    for engine in engines:
        engine.dispose()
