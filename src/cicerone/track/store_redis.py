"""Impression/click log in Redis. History and eval stay on the output store."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from cicerone.config.constants import ConfigError
from cicerone.config.settings import IOSettings
from cicerone.io.recommendation_schema import USER_COLUMN

_REDIS_TIMEOUT_SECONDS = 5.0
_REDIS_EXTRA = "install with: pip install 'cicerone-recommender[redis]'"
_HDEL_BATCH = 1000
_CREDENTIAL_WORDS = ("password", "secret", "token", "access_key", "credential")


def default_redis_key_prefix(output: IOSettings) -> str:
    """Prefix from the ``[output]`` location, so deployments with different outputs never share keys."""
    location = [output.kind]
    for key in sorted(output.options):
        if any(word in key for word in _CREDENTIAL_WORDS):
            continue
        value = output.options[key]
        # Credentials are not part of the location; rotating them must not move the log.
        location.append(f"{key}={_without_credentials(str(value)) if key.endswith('_url') else value}")
    digest = hashlib.sha256("\0".join(location).encode()).hexdigest()[:16]
    return f"cicerone:{digest}"


def _without_credentials(url: str) -> str:
    try:
        return make_url(url).set(username=None, password=None, query={}).render_as_string()
    except ArgumentError:
        return url.rsplit("@", 1)[-1].split("?", 1)[0]


def _parse_line(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _event_key(row: dict[str, Any], line: str) -> str:
    event_id = str(row.get("event_id") or "")
    return event_id or "sha256:" + hashlib.sha256(line.encode()).hexdigest()


class RedisTrackLog:
    """Hash of event id to JSON row: one HSETNX per append, HDEL per erase."""

    def __init__(self, redis_url: str, *, key_prefix: str, client: Any | None = None):
        self._client = client if client is not None else _connect(redis_url)
        self._log_key = f"{key_prefix}:track"

    def append(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        lines = [json.dumps(row, separators=(",", ":")) for row in rows]
        # One MULTI/EXEC for the batch: a failure before EXEC stores none of it, and HSETNX keeps
        # a retry after a lost reply idempotent.
        with self._client.pipeline(transaction=True) as pipe:
            for row, line in zip(rows, lines, strict=True):
                pipe.hsetnx(self._log_key, _event_key(row, line), line)
            stored = pipe.execute()
        return [row for row, fresh in zip(rows, stored, strict=True) if fresh]

    def read(self) -> list[dict[str, Any]]:
        rows = [parsed for raw in self._client.hvals(self._log_key) if (parsed := _parse_line(str(raw)))]
        rows.sort(key=lambda row: (str(row.get("occurred_at") or ""), str(row.get("event_id") or "")))
        return rows

    def delete_user(self, user_id: str) -> int:
        # HDEL touches only this user's fields, so concurrent appends and other rows are never rewritten.
        fields = [
            str(field)
            for field, raw in self._client.hgetall(self._log_key).items()
            if (parsed := _parse_line(str(raw))) is not None
            and str(parsed.get(USER_COLUMN) or "").strip() == user_id
        ]
        removed = 0
        for start in range(0, len(fields), _HDEL_BATCH):
            removed += int(self._client.hdel(self._log_key, *fields[start : start + _HDEL_BATCH]) or 0)
        return removed


def _connect(redis_url: str) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise ConfigError(f"track.redis_url requires the redis package; {_REDIS_EXTRA}") from exc
    # No eager ping: a down Redis surfaces as a RedisError on first use, which callers handle.
    return redis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_timeout=_REDIS_TIMEOUT_SECONDS,
        socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
    )
