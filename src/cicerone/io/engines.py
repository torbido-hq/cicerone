"""Shared SQLAlchemy engines keyed by database URL."""

from __future__ import annotations

import threading
from collections import OrderedDict

import sqlalchemy
from sqlalchemy import Engine

_MAX_CACHED_ENGINES = 8
_engines: OrderedDict[str, Engine] = OrderedDict()
_engines_lock = threading.Lock()


def engine_for(database_url: str) -> Engine:
    with _engines_lock:
        engine = _engines.get(database_url)
        if engine is not None:
            _engines.move_to_end(database_url)
            return engine
        engine = sqlalchemy.create_engine(database_url, pool_pre_ping=True)
        _engines[database_url] = engine
        while len(_engines) > _MAX_CACHED_ENGINES:
            _url, old = _engines.popitem(last=False)
            old.dispose()
        return engine


def dispose_engines() -> None:
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
    for engine in engines:
        engine.dispose()
