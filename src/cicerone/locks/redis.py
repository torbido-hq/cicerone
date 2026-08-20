"""Redis SET NX PX lock backend for multi-replica schedulers."""

from __future__ import annotations

import logging
import threading
import uuid

from cicerone.config.constants import DEFAULT_LOCK_KEY, ConfigError
from cicerone.locks.keys import REDIS_LOCK_TTL_MS

logger = logging.getLogger(__name__)

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
