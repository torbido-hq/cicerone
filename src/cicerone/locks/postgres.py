"""Postgres advisory-lock backend for multi-replica schedulers."""

from __future__ import annotations

import logging
import threading

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


class PostgresAdvisoryLock:
    """``pg_try_advisory_lock`` on a connection held for the run duration."""

    def __init__(self, database_url: str, *, lock_key: str = DEFAULT_LOCK_KEY):
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Connection

        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._conn: Connection | None = None
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
            except Exception:
                conn.close()
                raise
            if not got:
                conn.close()
                return False
            self._conn = conn
            return True

    def owned(self) -> bool:
        from sqlalchemy import text

        with self._mutex:
            if self._conn is None:
                return False
            conn = self._conn
            try:
                return bool(
                    conn.execute(
                        text(_PG_ADVISORY_HELD_SELF),
                        {"k1": self._key1, "k2": self._key2},
                    ).scalar()
                )
            except Exception:
                logger.warning(
                    "Postgres advisory lock owned() probe failed; treating as lost",
                    exc_info=True,
                )
                return False

    def is_locked(self) -> bool:
        from sqlalchemy import text

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
                except Exception:
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
        except Exception:
            logger.exception("Postgres advisory lock is_locked() probe failed")
            raise

    def release(self) -> None:
        from sqlalchemy import text

        with self._mutex:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    text("SELECT pg_advisory_unlock(:k1, :k2)"),
                    {"k1": self._key1, "k2": self._key2},
                )
            except Exception:
                logger.exception("Failed to release Postgres advisory lock")
            finally:
                self._conn.close()
                self._conn = None
