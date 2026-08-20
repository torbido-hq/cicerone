"""Optional distributed lock backends for multi-replica schedulers.

Default single-instance exclusion is RunGuard's threading.Lock (no backend).
``postgres`` / ``redis`` are opt-in; clients are imported only when selected.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from typing import Protocol

from cicerone.config.constants import (
    DEFAULT_LOCK_KEY,
    DEFAULT_LOCK_TTL_SECONDS,
    ConfigError,
)
from cicerone.config.lock_url import resolve_postgres_lock_url
from cicerone.config.settings import Settings

logger = logging.getLogger(__name__)

# Long enough for a full train; abandoned Redis holders expire eventually.
REDIS_LOCK_KEY = DEFAULT_LOCK_KEY
REDIS_LOCK_TTL_MS = DEFAULT_LOCK_TTL_SECONDS * 1000
_PG_LOCK_DIGEST = hashlib.sha256(DEFAULT_LOCK_KEY.encode()).digest()
PG_ADVISORY_KEY1 = int.from_bytes(_PG_LOCK_DIGEST[:4], "big") & 0x7FFFFFFF
PG_ADVISORY_KEY2 = int.from_bytes(_PG_LOCK_DIGEST[4:8], "big") & 0x7FFFFFFF

_REDIS_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

_REDIS_REFRESH_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""


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


class LockLostError(RuntimeError):
    """Lease expired or was stolen before a fenced write."""


class LockBackend(Protocol):
    def acquire(self) -> bool: ...

    def release(self) -> None: ...

    def owned(self) -> bool:
        """True when this instance still holds the lease (fencing)."""
        ...

    def is_locked(self) -> bool:
        """True when any process holds this key (probe; does not acquire)."""
        ...


def events_apply_lock_key(run_guard_key: str) -> str:
    """Lease key for incremental apply; distinct from the full-retrain lock."""
    return f"{run_guard_key}:events:apply"


def advisory_keys_from_lock_key(lock_key: str) -> tuple[int, int]:
    digest = hashlib.sha256(lock_key.encode()).digest()
    return (
        int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
        int.from_bytes(digest[4:8], "big") & 0x7FFFFFFF,
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


class RedisLock:
    """``SET NX PX`` lock; TTL is refreshed while held so long jobs stay exclusive."""

    def __init__(
        self,
        redis_url: str,
        *,
        key: str = DEFAULT_LOCK_KEY,
        ttl_ms: int = REDIS_LOCK_TTL_MS,
        refresh_interval_ms: int | None = None,
    ):
        try:
            import redis
        except ImportError as exc:
            raise ConfigError(
                'lock_backend = "redis" requires the redis package; '
                "install with: pip install 'cicerone-recommender[redis]'"
            ) from exc
        self._client = redis.Redis.from_url(redis_url)
        self._key = key
        self._ttl_ms = ttl_ms
        self._refresh_interval_ms = (
            max(ttl_ms // 2, 1) if refresh_interval_ms is None else max(refresh_interval_ms, 1)
        )
        self._token = str(uuid.uuid4())
        self._held = False
        self._mutex = threading.Lock()
        self._release_script = self._client.register_script(_REDIS_RELEASE_SCRIPT)
        self._refresh_script = self._client.register_script(_REDIS_REFRESH_SCRIPT)
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    def _mark_lost(self) -> None:
        """Clear local hold state so a later acquire() can succeed after TTL loss."""
        with self._mutex:
            self._held = False
            self._token = str(uuid.uuid4())
        self._stop_refresh.set()
        # Drop the handle so acquire() can start a new refresher (we may be that thread).
        self._refresh_thread = None

    def _start_refresh(self) -> None:
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        self._stop_refresh.clear()

        def _run() -> None:
            while not self._stop_refresh.wait(self._refresh_interval_ms / 1000.0):
                with self._mutex:
                    if not self._held:
                        break
                    token = self._token
                try:
                    if not self._refresh_script(keys=[self._key], args=[token, self._ttl_ms]):
                        # Intentional release sets stop before clearing hold; skip _mark_lost.
                        if self._stop_refresh.is_set():
                            break
                        self._mark_lost()
                        break
                except Exception:
                    if self._stop_refresh.is_set():
                        break
                    logger.exception("Failed to refresh Redis lock TTL")
                    self._mark_lost()
                    break

        self._refresh_thread = threading.Thread(
            target=_run,
            name=f"cicerone-redis-lock-refresh-{self._key}",
            daemon=True,
        )
        self._refresh_thread.start()

    def _stop_refresh_thread(self) -> None:
        thread = self._refresh_thread
        if thread is None:
            return
        self._stop_refresh.set()
        # Brief wait for a refresh already past wait(); stop flag skips _mark_lost.
        # Cap at 250ms so a stuck Redis Lua call cannot delay release() for a full interval.
        thread.join(timeout=0.25)
        self._refresh_thread = None

    def acquire(self) -> bool:
        with self._mutex:
            if self._held:
                return False
            token = self._token
        ok = bool(self._client.set(self._key, token, nx=True, px=self._ttl_ms))
        if not ok:
            return False
        with self._mutex:
            self._held = True
        self._start_refresh()
        return True

    def owned(self) -> bool:
        with self._mutex:
            if not self._held:
                return False
            token = self._token
        try:
            value = self._client.get(self._key)
        except Exception:
            logger.warning("Redis lock owned() probe failed; treating as lost", exc_info=True)
            return False
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value == token

    def is_locked(self) -> bool:
        return bool(self._client.exists(self._key))

    def release(self) -> None:
        self._stop_refresh_thread()
        with self._mutex:
            if not self._held:
                return
            token = self._token
            self._held = False
            self._token = str(uuid.uuid4())
        try:
            self._release_script(keys=[self._key], args=[token])
        except Exception:
            logger.exception("Failed to release Redis lock")


def build_lock_backend(
    settings: Settings,
    *,
    lock_key: str | None = None,
    ttl_seconds: float | None = None,
) -> LockBackend:
    """Build a distributed lock backend (``postgres`` / ``redis`` only).

    ``lock_backend`` and required URLs must already be validated at config load.
    Callers must not invoke this for ``in_process``.
    ``lock_key`` overrides ``settings.trigger.lock_key`` (events apply lease).
    ``ttl_seconds`` overrides Redis TTL (Postgres advisory locks have none).
    """
    backend = settings.trigger.lock_backend
    key = settings.trigger.lock_key if lock_key is None else lock_key
    if backend == "postgres":
        url = resolve_postgres_lock_url(settings)
        assert url is not None, "postgres lock URL should be validated at config load"
        return PostgresAdvisoryLock(url, lock_key=key)
    if backend == "redis":
        redis_url = settings.trigger.redis_url
        assert redis_url is not None, "redis_url should be validated at config load"
        ttl = settings.trigger.lock_ttl_seconds if ttl_seconds is None else ttl_seconds
        return RedisLock(
            redis_url,
            key=key,
            ttl_ms=int(ttl * 1000),
        )
    raise AssertionError(f"build_lock_backend is only for distributed backends, got {backend!r}")
