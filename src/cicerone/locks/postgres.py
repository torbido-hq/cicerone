"""Postgres advisory-lock backend for multi-replica schedulers."""

from __future__ import annotations

import logging
import threading
from functools import cache
from typing import TYPE_CHECKING

from cicerone.config.constants import DEFAULT_LOCK_KEY
from cicerone.locks.keys import advisory_keys_from_lock_key

logger = logging.getLogger(__name__)

# Two-key advisory: classid/objid = key1/key2, objsubid = 2 (not the 64-bit form).
_PG_ADVISORY_HELD_ANY = (
    "SELECT EXISTS ("
    "SELECT 1 FROM pg_locks "
    "WHERE locktype = 'advisory' AND classid = :k1 AND objid = :k2 "
    "AND objsubid = 2 AND granted)"
)
_PG_ADVISORY_HELD_SELF = (
    "SELECT EXISTS ("
    "SELECT 1 FROM pg_locks "
    "WHERE locktype = 'advisory' AND classid = :k1 AND objid = :k2 "
    "AND objsubid = 2 AND granted AND pid = pg_backend_pid())"
)


if TYPE_CHECKING:
    from sqlalchemy import Table


@cache
def _input_markers() -> Table:
    """Input change each lock key's last poll-triggered run started from."""
    from sqlalchemy import Column, MetaData, Table, Text

    return Table(
        "cicerone_input_markers",
        MetaData(),
        Column("lock_key", Text, primary_key=True),
        Column("marker", Text, nullable=False),
    )


class PostgresAdvisoryLock:
    """``pg_try_advisory_lock`` on a connection held for the run duration."""

    def __init__(self, database_url: str, *, lock_key: str = DEFAULT_LOCK_KEY):
        from sqlalchemy.engine import Connection

        from cicerone.io.engines import engine_for

        self._engine = engine_for(database_url)
        self._conn: Connection | None = None
        self._lock_key = lock_key
        self._key1, self._key2 = advisory_keys_from_lock_key(lock_key)
        self._mutex = threading.Lock()

    def acquire(self) -> bool:
        from sqlalchemy import text

        with self._mutex:
            if self._conn is not None:
                return False
            conn = self._engine.connect()
            try:
                got = bool(
                    conn.execute(
                        text("SELECT pg_try_advisory_lock(:k1, :k2)"),
                        {"k1": self._key1, "k2": self._key2},
                    ).scalar()
                )
                if not got:
                    return False
                self._conn = conn
                return True
            finally:
                if self._conn is not conn:
                    conn.close()

    def owned(self, generation: int | None = None) -> bool:
        del generation
        from sqlalchemy import text
        from sqlalchemy.exc import SQLAlchemyError

        with self._mutex:
            if self._conn is None:
                return False
            conn = self._conn
            try:
                held = bool(
                    conn.execute(
                        text(_PG_ADVISORY_HELD_SELF),
                        {"k1": self._key1, "k2": self._key2},
                    ).scalar()
                )
            except SQLAlchemyError:
                logger.warning(
                    "Postgres advisory lock owned() probe failed; treating as lost",
                    exc_info=True,
                )
                held = False
            if not held:
                # The session no longer holds the lock: drop it so acquire() can open a fresh one.
                self._conn = None
                conn.invalidate()
                conn.close()
            return held

    def last_input_marker(self) -> str | None:
        from sqlalchemy import select

        markers = _input_markers()
        with self._engine.connect() as conn:
            markers.create(conn, checkfirst=True)
            conn.commit()
            return conn.execute(select(markers.c.marker).where(markers.c.lock_key == self._lock_key)).scalar()

    def set_input_marker(self, marker: str) -> None:
        from sqlalchemy.dialects.postgresql import insert

        row = {"lock_key": self._lock_key, "marker": marker}
        markers = _input_markers()
        with self._engine.begin() as conn:
            markers.create(conn, checkfirst=True)
            upsert = insert(markers).values(row)
            conn.execute(upsert.on_conflict_do_update(index_elements=["lock_key"], set_=row))

    def is_locked(self) -> bool:
        from sqlalchemy import text
        from sqlalchemy.exc import SQLAlchemyError

        with self._mutex:
            conn = self._conn
            if conn is not None:
                try:
                    return bool(
                        conn.execute(
                            text(_PG_ADVISORY_HELD_ANY),
                            {"k1": self._key1, "k2": self._key2},
                        ).scalar()
                    )
                except SQLAlchemyError:
                    logger.warning(
                        "Postgres advisory lock is_locked() on held connection failed; "
                        "retrying with a new session",
                        exc_info=True,
                    )
        try:
            with self._engine.connect() as probe:
                return bool(
                    probe.execute(
                        text(_PG_ADVISORY_HELD_ANY),
                        {"k1": self._key1, "k2": self._key2},
                    ).scalar()
                )
        except SQLAlchemyError:
            logger.exception("Postgres advisory lock is_locked() probe failed")
            raise

    def release(self) -> None:
        from sqlalchemy import text
        from sqlalchemy.exc import SQLAlchemyError

        with self._mutex:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    text("SELECT pg_advisory_unlock(:k1, :k2)"),
                    {"k1": self._key1, "k2": self._key2},
                )
            except SQLAlchemyError:
                logger.exception("Failed to release Postgres advisory lock")
            finally:
                self._conn.close()
                self._conn = None
