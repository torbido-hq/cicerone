"""Impression/click log in Redis. History and eval stay on the output store."""

from __future__ import annotations

import json
from typing import Any

from cicerone.config.constants import DEFAULT_TRACK_REDIS_KEY_PREFIX, ConfigError
from cicerone.io.recommendation_schema import USER_COLUMN

DEFAULT_REDIS_KEY_PREFIX = DEFAULT_TRACK_REDIS_KEY_PREFIX
_REDIS_TIMEOUT_SECONDS = 5.0
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
        # SADD is the atomic claim, so distinct ids never contend. A failed push releases
        # the claim so a retry can store the row; only a crash in between strands an id.
        if not self._client.sadd(self._ids_key, event_id):
            return False
        try:
            self._client.rpush(self._log_key, line)
        except BaseException as exc:
            try:
                self._client.srem(self._ids_key, event_id)
            except Exception:
                exc.add_note(f"event id {event_id!r} stays claimed; a retry will skip it")
            raise
        return True

    def read(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in self._client.lrange(self._log_key, 0, -1):
            parsed = _parse_line(str(raw))
            if parsed is not None:
                rows.append(parsed)
        return rows

    def delete_user(self, user_id: str) -> int:
        # LREM removes the user's exact lines in place: other rows and concurrent appends
        # are never rewritten, so a failed command cannot drop them.
        matched: dict[str, str] = {}
        for raw in self._client.lrange(self._log_key, 0, -1):
            line = str(raw)
            parsed = _parse_line(line)
            if parsed is not None and str(parsed.get(USER_COLUMN) or "").strip() == user_id:
                matched[line] = str(parsed.get("event_id") or "")
        # Release the ids first: a failure part-way then leaves rows for a rerun, never a
        # claim that would make a later append of the same event skip its row.
        event_ids = [event_id for event_id in matched.values() if event_id]
        if event_ids:
            self._client.srem(self._ids_key, *event_ids)
        removed = 0
        for line in matched:
            removed += int(self._client.lrem(self._log_key, 0, line) or 0)
        return removed


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
