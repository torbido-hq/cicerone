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
from cicerone.config.lock_url import require_postgres_lock_url
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


class LockBackend(Protocol):
    def acquire(self) -> bool: ...

    def release(self) -> None: ...


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

    def acquire(self) -> bool:
        from sqlalchemy import text

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

    def release(self) -> None:
        from sqlalchemy import text

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
                "install with: pip install -r requirements-redis.txt"
            ) from exc
        self._client = redis.Redis.from_url(redis_url)
        self._key = key
        self._ttl_ms = ttl_ms
        self._refresh_interval_ms = (
            max(ttl_ms // 2, 1) if refresh_interval_ms is None else max(refresh_interval_ms, 1)
        )
        self._token = str(uuid.uuid4())
        self._held = False
        self._release_script = self._client.register_script(_REDIS_RELEASE_SCRIPT)
        self._refresh_script = self._client.register_script(_REDIS_REFRESH_SCRIPT)
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    def _start_refresh(self) -> None:
        if self._refresh_thread is not None:
            return
        self._stop_refresh.clear()

        def _run() -> None:
            while not self._stop_refresh.wait(self._refresh_interval_ms / 1000.0):
                if not self._held:
                    break
                try:
                    if not self._refresh_script(keys=[self._key], args=[self._token, self._ttl_ms]):
                        break
                except Exception:
                    logger.exception("Failed to refresh Redis lock TTL")
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
        thread.join(timeout=1.0)
        self._refresh_thread = None

    def acquire(self) -> bool:
        if self._held:
            return False
        ok = bool(self._client.set(self._key, self._token, nx=True, px=self._ttl_ms))
        self._held = ok
        if ok:
            self._start_refresh()
        return ok

    def release(self) -> None:
        if not self._held:
            return
        self._stop_refresh_thread()
        try:
            self._release_script(keys=[self._key], args=[self._token])
        except Exception:
            logger.exception("Failed to release Redis lock")
        finally:
            self._held = False


def build_lock_backend(settings: Settings) -> LockBackend:
    """Build a distributed lock backend (``postgres`` / ``redis`` only).

    ``lock_backend`` must already be validated at config load. Callers must
    not invoke this for ``in_process`` — that path uses RunGuard with no backend.
    """
    backend = settings.trigger.lock_backend
    lock_key = settings.trigger.lock_key
    if backend == "postgres":
        return PostgresAdvisoryLock(require_postgres_lock_url(settings), lock_key=lock_key)
    if backend == "redis":
        redis_url = settings.trigger.redis_url
        if not redis_url:
            raise ConfigError('lock_backend = "redis" requires job.trigger.redis_url')
        return RedisLock(
            redis_url,
            key=lock_key,
            ttl_ms=int(settings.trigger.lock_ttl_seconds * 1000),
        )
    raise AssertionError(f"build_lock_backend is only for distributed backends, got {backend!r}")
