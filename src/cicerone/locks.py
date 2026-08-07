"""Optional distributed lock backends for the scheduler RunGuard.

Default ``in_process`` needs no extra config or deps. ``postgres`` and ``redis``
are opt-in for multi-replica schedulers; their clients are imported only when
selected.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Protocol

from cicerone.config.constants import LOCK_BACKENDS, ConfigError
from cicerone.config.settings import Settings

logger = logging.getLogger(__name__)

# Session-level advisory lock keys (stable across processes / restarts).
_PG_LOCK_DIGEST = hashlib.sha256(b"cicerone:scheduler:run_guard").digest()
PG_ADVISORY_KEY1 = int.from_bytes(_PG_LOCK_DIGEST[:4], "big") & 0x7FFFFFFF
PG_ADVISORY_KEY2 = int.from_bytes(_PG_LOCK_DIGEST[4:8], "big") & 0x7FFFFFFF

REDIS_LOCK_KEY = "cicerone:scheduler:run_guard"
# Long enough for a full train; abandoned holders expire eventually.
REDIS_LOCK_TTL_MS = 24 * 60 * 60 * 1000

_REDIS_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class LockBackend(Protocol):
    def acquire(self) -> bool: ...

    def release(self) -> None: ...


class InProcessLock:
    """No-op: per-process exclusion stays in RunGuard's threading.Lock."""

    def acquire(self) -> bool:
        return True

    def release(self) -> None:
        return None


class PostgresAdvisoryLock:
    """``pg_try_advisory_lock`` on a connection held for the run duration."""

    def __init__(self, database_url: str):
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Connection

        self._engine = create_engine(database_url, pool_pre_ping=True)
        self._conn: Connection | None = None

    def acquire(self) -> bool:
        from sqlalchemy import text

        if self._conn is not None:
            return False
        conn = self._engine.connect()
        try:
            got = bool(
                conn.execute(
                    text("SELECT pg_try_advisory_lock(:k1, :k2)"),
                    {"k1": PG_ADVISORY_KEY1, "k2": PG_ADVISORY_KEY2},
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
                {"k1": PG_ADVISORY_KEY1, "k2": PG_ADVISORY_KEY2},
            )
        except Exception:
            logger.exception("Failed to release Postgres advisory lock")
        finally:
            self._conn.close()
            self._conn = None


class RedisLock:
    """``SET key token NX PX ttl`` with compare-and-delete release."""

    def __init__(
        self,
        redis_url: str,
        *,
        key: str = REDIS_LOCK_KEY,
        ttl_ms: int = REDIS_LOCK_TTL_MS,
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
        self._token = str(uuid.uuid4())
        self._held = False

    def acquire(self) -> bool:
        if self._held:
            return False
        ok = bool(self._client.set(self._key, self._token, nx=True, px=self._ttl_ms))
        self._held = ok
        return ok

    def release(self) -> None:
        if not self._held:
            return
        try:
            self._client.eval(_REDIS_RELEASE_SCRIPT, 1, self._key, self._token)
        except Exception:
            logger.exception("Failed to release Redis lock")
        finally:
            self._held = False


def resolve_postgres_lock_url(settings: Settings) -> str | None:
    """Explicit trigger URL wins; otherwise reuse ``[output]`` when ``kind = "db"``."""
    if settings.trigger.postgres_url:
        return settings.trigger.postgres_url
    if settings.output.kind == "db":
        url = settings.output.options.get("database_url")
        return str(url) if url else None
    return None


def build_lock_backend(settings: Settings) -> LockBackend:
    backend = settings.trigger.lock_backend
    if backend == "in_process":
        return InProcessLock()
    if backend == "postgres":
        url = resolve_postgres_lock_url(settings)
        if not url:
            raise ConfigError(
                'lock_backend = "postgres" needs a database URL: set '
                '[job.trigger].postgres_url, or use [output].kind = "db" with '
                "[output.options].database_url"
            )
        return PostgresAdvisoryLock(url)
    if backend == "redis":
        redis_url = settings.trigger.redis_url
        if not redis_url:
            raise ConfigError('lock_backend = "redis" requires job.trigger.redis_url')
        return RedisLock(redis_url)
    raise ConfigError(f"job.trigger.lock_backend must be one of {list(LOCK_BACKENDS)}, got {backend!r}")
