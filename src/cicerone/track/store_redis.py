"""Impression/click log in Redis. History and eval stay on the output store."""

from __future__ import annotations

import json
from typing import Any

from cicerone.config.constants import DEFAULT_TRACK_REDIS_KEY_PREFIX, ConfigError
from cicerone.io.recommendation_schema import USER_COLUMN

try:
    from redis.exceptions import WatchError
except ImportError:

    class WatchError(Exception):  # type: ignore[no-redef]
        pass


DEFAULT_REDIS_KEY_PREFIX = DEFAULT_TRACK_REDIS_KEY_PREFIX
_REDIS_TIMEOUT_SECONDS = 5.0
_WATCH_RETRIES = 32
_REDIS_EXTRA = "install with: pip install 'cicerone-recommender[redis]'"


def _parse_line(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class RedisTrackLog:
    """Append-only JSON list plus an event-id set."""

    def __init__(
        self, redis_url: str, *, client: Any | None = None, key_prefix: str = DEFAULT_REDIS_KEY_PREFIX
    ):
        self._client = client if client is not None else _connect(redis_url)
        # A prefix per deployment keeps logs apart when they share one Redis database.
        self._log_key = f"{key_prefix}:track"
        self._ids_key = f"{key_prefix}:track:ids"

    def append(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fresh: list[dict[str, Any]] = []
        for row in rows:
            if self._append_one(row):
                fresh.append(row)
        return fresh

    def _append_one(self, row: dict[str, Any]) -> bool:
        line = json.dumps(row, separators=(",", ":"))
        event_id = str(row.get("event_id") or "")
        if not event_id:
            self._client.rpush(self._log_key, line)
            return True
        # The id claim and the push commit together, so a failed push cannot strand the id.
        with self._client.pipeline() as pipe:
            for _attempt in range(_WATCH_RETRIES):
                try:
                    pipe.watch(self._ids_key)
                    if pipe.sismember(self._ids_key, event_id):
                        return False
                    pipe.multi()
                    pipe.sadd(self._ids_key, event_id)
                    pipe.rpush(self._log_key, line)
                    pipe.execute()
                    return True
                except WatchError:
                    continue
        raise WatchError(f"track log kept changing; gave up after {_WATCH_RETRIES} attempts")

    def read(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in self._client.lrange(self._log_key, 0, -1):
            parsed = _parse_line(str(raw))
            if parsed is not None:
                rows.append(parsed)
        return rows

    def delete_user(self, user_id: str) -> int:
        # WATCH the list so an append between the read and the rewrite retries instead of being lost.
        with self._client.pipeline() as pipe:
            for _attempt in range(_WATCH_RETRIES):
                try:
                    pipe.watch(self._log_key)
                    kept: list[str] = []
                    removed_ids: list[str] = []
                    removed = 0
                    for raw in pipe.lrange(self._log_key, 0, -1):
                        line = str(raw)
                        parsed = _parse_line(line)
                        if parsed is not None and str(parsed.get(USER_COLUMN) or "").strip() == user_id:
                            removed += 1
                            event_id = str(parsed.get("event_id") or "")
                            if event_id:
                                removed_ids.append(event_id)
                            continue
                        kept.append(line)
                    if not removed:
                        return 0
                    pipe.multi()
                    pipe.delete(self._log_key)
                    if kept:
                        pipe.rpush(self._log_key, *kept)
                    if removed_ids:
                        pipe.srem(self._ids_key, *removed_ids)
                    pipe.execute()
                    return removed
                except WatchError:
                    continue
        raise WatchError(f"track log kept changing; gave up after {_WATCH_RETRIES} attempts")


def _connect(redis_url: str) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise ConfigError(f"track.redis_url requires the redis package; {_REDIS_EXTRA}") from exc
    client = redis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_timeout=_REDIS_TIMEOUT_SECONDS,
        socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
    )
    try:
        client.ping()
    except redis.RedisError as exc:
        raise ConfigError(f"track.redis_url is unreachable: {exc}") from exc
    return client
