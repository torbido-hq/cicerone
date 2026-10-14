"""Impression/click log in Redis. History and eval stay on the output store."""

from __future__ import annotations

import json
from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.io.recommendation_schema import USER_COLUMN

TRACK_LOG_KEY = "cicerone:track"
TRACK_IDS_KEY = "cicerone:track:ids"
_REDIS_EXTRA = "install with: pip install 'cicerone-recommender[redis]'"


def _parse_line(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class RedisTrackLog:
    """Append-only JSON list plus an event-id set."""

    def __init__(self, redis_url: str, *, client: Any | None = None):
        self._client = client if client is not None else _connect(redis_url)

    def append(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fresh: list[dict[str, Any]] = []
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if event_id and not self._client.sadd(TRACK_IDS_KEY, event_id):
                continue
            self._client.rpush(TRACK_LOG_KEY, json.dumps(row, separators=(",", ":")))
            fresh.append(row)
        return fresh

    def read(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in self._client.lrange(TRACK_LOG_KEY, 0, -1):
            parsed = _parse_line(str(raw))
            if parsed is not None:
                rows.append(parsed)
        return rows

    def delete_user(self, user_id: str) -> int:
        kept: list[str] = []
        removed = 0
        removed_ids: list[str] = []
        for raw in self._client.lrange(TRACK_LOG_KEY, 0, -1):
            line = str(raw)
            parsed = _parse_line(line)
            if parsed is not None and str(parsed.get(USER_COLUMN) or "").strip() == user_id:
                removed += 1
                event_id = str(parsed.get("event_id") or "")
                if event_id:
                    removed_ids.append(event_id)
                continue
            kept.append(line)
        pipe = self._client.pipeline()
        pipe.delete(TRACK_LOG_KEY)
        if kept:
            pipe.rpush(TRACK_LOG_KEY, *kept)
        if removed_ids:
            pipe.srem(TRACK_IDS_KEY, *removed_ids)
        pipe.execute()
        return removed


def _connect(redis_url: str) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise ConfigError(f"track.redis_url requires the redis package; {_REDIS_EXTRA}") from exc
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        client.ping()
    except redis.RedisError as exc:
        raise ConfigError(f"track.redis_url is unreachable: {exc}") from exc
    return client
