"""Shared SQLAlchemy engines keyed by database URL."""

from __future__ import annotations

import threading
from typing import Any

import sqlalchemy
from sqlalchemy import Engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

EngineKey = str | tuple[str, int]

_engines: dict[EngineKey, Engine] = {}
_checkouts: dict[EngineKey, int] = {}
_memory_options: dict[EngineKey, dict[str, Any]] = {}
_engines_lock = threading.Lock()


def _is_memory_sqlite(database_url: str) -> bool:
    parsed = make_url(database_url)
    return parsed.get_backend_name() == "sqlite" and parsed.database in (None, "", ":memory:")


def _engine_key(database_url: str, options: dict[str, Any] | None) -> EngineKey:
    if options is not None and _is_memory_sqlite(database_url):
        return database_url, id(options)
    return database_url


def engine_for(database_url: str, *, options: dict[str, Any] | None = None) -> Engine:
    key = _engine_key(database_url, options)
    with _engines_lock:
        engine = _engines.get(key)
        if engine is None:
            kwargs: dict[str, Any] = {"pool_pre_ping": True}
            if _is_memory_sqlite(database_url):
                kwargs["poolclass"] = StaticPool
                kwargs["connect_args"] = {"check_same_thread": False}
            engine = sqlalchemy.create_engine(database_url, **kwargs)
            _engines[key] = engine
            _checkouts[key] = 0
            if options is not None and isinstance(key, tuple):
                _memory_options[key] = options
        _checkouts[key] += 1
        return engine


def release_engine(database_url: str, *, options: dict[str, Any] | None = None) -> None:
    key = _engine_key(database_url, options)
    with _engines_lock:
        remaining = _checkouts.get(key)
        if remaining is None:
            return
        remaining -= 1
        if remaining > 0:
            _checkouts[key] = remaining
            return
        engine = _engines.pop(key, None)
        _checkouts.pop(key, None)
        _memory_options.pop(key, None)
    if engine is not None:
        engine.dispose()


def dispose_engines() -> None:
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
        _checkouts.clear()
        _memory_options.clear()
    for engine in engines:
        engine.dispose()
