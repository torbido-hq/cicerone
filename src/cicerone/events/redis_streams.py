"""Redis Streams EventSource (consumer group + XACK)."""

from __future__ import annotations

import logging
import socket
import threading
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from cicerone.config import ConfigError
from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.normalize import EventNormalizeError, normalize_event

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_MS = 0
_DEFAULT_CLAIM_IDLE_MS = 300_000
_DEFAULT_GROUP_START_ID = "0-0"


def validate_redis_stream_options(options: dict[str, Any]) -> None:
    """Validate ``events.options`` for ``kind=redis_streams``."""
    for key in ("redis_url", "stream", "consumer_group"):
        value = options.get(key)
        if value in (None, "") or (isinstance(value, str) and not value.strip()):
            raise ConfigError(f'events.options.{key} is required when events.kind = "redis_streams"')
    if "consumer_name" in options and str(options["consumer_name"]).strip() == "":
        raise ConfigError("events.options.consumer_name must be non-empty when set")
    if "group_start_id" in options and str(options["group_start_id"]).strip() == "":
        raise ConfigError("events.options.group_start_id must be non-empty when set")
    block_ms = _as_int(options, "block_ms", _DEFAULT_BLOCK_MS)
    if block_ms < 0:
        raise ConfigError(f"events.options.block_ms must be >= 0, got {block_ms}")
    claim_idle_ms = _as_int(options, "claim_idle_ms", _DEFAULT_CLAIM_IDLE_MS)
    if claim_idle_ms < 1:
        raise ConfigError(f"events.options.claim_idle_ms must be >= 1, got {claim_idle_ms}")


def _as_int(options: dict[str, Any], key: str, default: int) -> int:
    raw = options.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"events.options.{key} must be an integer, got {raw!r}") from exc


def _require_str(options: dict[str, Any], key: str) -> str:
    value = str(options[key]).strip()
    if not value:
        raise ConfigError(f'events.options.{key} is required when events.kind = "redis_streams"')
    return value


class RedisStreamsEventSource(EventSource):
    """Consume flat stream hashes via a consumer group; ack with ``XACK``."""

    def __init__(self, options: dict[str, Any]):
        validate_redis_stream_options(options)
        self._redis_url = _require_str(options, "redis_url")
        self._stream = _require_str(options, "stream")
        self._group = _require_str(options, "consumer_group")
        raw_name = options.get("consumer_name")
        self._consumer = (
            str(raw_name).strip() if raw_name not in (None, "") else socket.gethostname() or "cicerone"
        )
        start = options.get("group_start_id", _DEFAULT_GROUP_START_ID)
        self._group_start_id = str(start).strip() or _DEFAULT_GROUP_START_ID
        self._block_ms = _as_int(options, "block_ms", _DEFAULT_BLOCK_MS)
        self._claim_idle_ms = _as_int(options, "claim_idle_ms", _DEFAULT_CLAIM_IDLE_MS)

        self._client: Any | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._pending: deque[NormalizedEvent] = deque()
        self._pending_ids: set[str] = set()
        self._in_flight: set[str] = set()
        self._entry_ids: dict[str, str] = {}  # event_id → stream entry id (until XACK)
        self._held_entries: set[str] = set()  # stream entry ids we already track
        self._last_event_at: datetime | None = None
        self._claim_cursor = "0-0"

    def connect(self) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ConfigError(
                'events.kind = "redis_streams" requires the redis package; '
                "install with: pip install 'cicerone-recommender[redis]'"
            ) from exc

        client = redis.Redis.from_url(self._redis_url, decode_responses=True)
        try:
            client.ping()
        except Exception as exc:
            raise ConfigError(f"events.options.redis_url is unreachable: {exc}") from exc

        try:
            client.xgroup_create(
                self._stream,
                self._group,
                id=self._group_start_id,
                mkstream=True,
            )
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

        with self._lock:
            previous = self._client
            self._client = client
            self._connected = True
            self._claim_cursor = "0-0"
        if previous is not None and previous is not client:
            try:
                previous.close()
            except Exception:
                logger.exception("Failed to close previous Redis Streams client")

    def close(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._connected = False
            self._pending.clear()
            self._pending_ids.clear()
            self._in_flight.clear()
            self._entry_ids.clear()
            self._held_entries.clear()
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.exception("Redis Streams client close failed")

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        client = self._require_client()
        out: list[NormalizedEvent] = []

        with self._lock:
            while self._pending and len(out) < max_events:
                event = self._pending.popleft()
                self._pending_ids.discard(event.event_id)
                self._in_flight.add(event.event_id)
                out.append(event)

        remaining = max_events - len(out)
        if remaining > 0:
            out.extend(self._claim_idle(client, remaining))
            remaining = max_events - len(out)
        if remaining > 0:
            out.extend(self._read_new(client, remaining))

        if out:
            newest = max(event.occurred_at for event in out)
            with self._lock:
                self._last_event_at = newest
        return out

    def ack(self, event_ids: Sequence[str]) -> None:
        if not event_ids:
            return
        client = self._require_client()
        with self._lock:
            resolved: list[tuple[str, str]] = []
            for event_id in event_ids:
                eid = str(event_id)
                entry_id = self._entry_ids.get(eid)
                if entry_id is not None:
                    resolved.append((eid, entry_id))
        if not resolved:
            return
        # XACK before dropping local maps so a failed ack can still nack/retry.
        client.xack(self._stream, self._group, *(entry_id for _, entry_id in resolved))
        with self._lock:
            for eid, entry_id in resolved:
                self._entry_ids.pop(eid, None)
                self._held_entries.discard(entry_id)
                self._in_flight.discard(eid)
                self._pending_ids.discard(eid)

    def nack(self, events: Sequence[NormalizedEvent]) -> None:
        if not events:
            return
        with self._lock:
            for event in reversed(list(events)):
                if event.event_id not in self._entry_ids:
                    continue
                self._in_flight.discard(event.event_id)
                if event.event_id in self._pending_ids:
                    continue
                self._pending.appendleft(event)
                self._pending_ids.add(event.event_id)

    def heartbeat(self, events: Sequence[NormalizedEvent]) -> None:
        if not events:
            return
        client = self._require_client()
        with self._lock:
            entry_ids = [
                self._entry_ids[event.event_id] for event in events if event.event_id in self._entry_ids
            ]
        if not entry_ids:
            return
        try:
            client.xclaim(
                self._stream,
                self._group,
                self._consumer,
                min_idle_time=0,
                message_ids=entry_ids,
            )
        except Exception:
            logger.exception("Redis Streams heartbeat XCLAIM failed")

    def health(self) -> EventSourceHealth:
        with self._lock:
            connected = self._connected
            client = self._client
            local_held = len(self._pending_ids) + len(self._in_flight)
            last_event_at = self._last_event_at
        if not connected or client is None:
            return EventSourceHealth(connected=False, lag=None, last_event_at=last_event_at)

        pending_count: int | None
        group_lag: int | None
        try:
            pending_count = int(client.xpending(self._stream, self._group)["pending"])
        except Exception:
            logger.exception("Redis Streams XPENDING failed")
            pending_count = None
        try:
            group_lag = self._group_lag(client)
        except Exception:
            logger.exception("Redis Streams XINFO GROUPS lag failed")
            group_lag = None

        if pending_count is None and group_lag is None:
            lag: int | None = local_held if local_held else None
        else:
            lag = (pending_count or 0) + (group_lag or 0)

        return EventSourceHealth(
            connected=True,
            lag=lag,
            last_event_at=last_event_at,
            detail=f"stream={self._stream} group={self._group} consumer={self._consumer}",
        )

    def _require_client(self) -> Any:
        with self._lock:
            if not self._connected or self._client is None:
                raise RuntimeError("RedisStreamsEventSource is not connected")
            return self._client

    def _group_lag(self, client: Any) -> int | None:
        groups = client.xinfo_groups(self._stream)
        for group in groups:
            name = group.get("name") or group.get(b"name")
            if str(name) != self._group:
                continue
            if "lag" in group:
                return int(group["lag"])
            if b"lag" in group:
                return int(group[b"lag"])
            return None
        return None

    def _claim_idle(self, client: Any, max_events: int) -> list[NormalizedEvent]:
        with self._lock:
            start_id = self._claim_cursor
        try:
            result = client.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                min_idle_time=self._claim_idle_ms,
                start_id=start_id,
                count=max_events,
            )
        except Exception:
            logger.exception("Redis Streams XAUTOCLAIM failed")
            return []

        next_id, entries = self._parse_autoclaim(result)
        if next_id is not None:
            with self._lock:
                self._claim_cursor = next_id
        return self._entries_to_events(client, entries)

    def _read_new(self, client: Any, max_events: int) -> list[NormalizedEvent]:
        try:
            raw = client.xreadgroup(
                self._group,
                self._consumer,
                streams={self._stream: ">"},
                count=max_events,
                block=self._block_ms,
            )
        except Exception:
            logger.exception("Redis Streams XREADGROUP failed")
            return []
        entries: list[tuple[str, dict[str, Any]]] = []
        for _stream_name, messages in raw or []:
            for entry_id, fields in messages:
                entries.append((str(entry_id), dict(fields)))
        return self._entries_to_events(client, entries)

    def _parse_autoclaim(self, result: Any) -> tuple[str | None, list[tuple[str, dict[str, Any]]]]:
        if not result:
            return None, []
        next_id = str(result[0]) if result[0] is not None else None
        messages = result[1] if len(result) > 1 else []
        entries = [(str(entry_id), dict(fields)) for entry_id, fields in messages or []]
        return next_id, entries

    def _entries_to_events(
        self,
        client: Any,
        entries: Sequence[tuple[str, dict[str, Any]]],
    ) -> list[NormalizedEvent]:
        events: list[NormalizedEvent] = []
        drop_ids: list[str] = []
        for entry_id, fields in entries:
            with self._lock:
                if entry_id in self._held_entries:
                    continue
            payload = {str(key): value for key, value in fields.items()}
            if payload.get("event_id") in (None, "") and payload.get("idempotency_key") in (None, ""):
                payload["event_id"] = entry_id
            try:
                event = normalize_event(payload)
            except EventNormalizeError as exc:
                logger.warning("Skipping invalid Redis Streams entry %s: %s", entry_id, exc)
                drop_ids.append(entry_id)
                continue
            with self._lock:
                existing_entry = self._entry_ids.get(event.event_id)
                if existing_entry is not None:
                    # Same logical event_id already held — drop the duplicate Redis entry.
                    logger.warning(
                        "Duplicate event_id %r on Redis entry %s (held %s); XACK duplicate",
                        event.event_id,
                        entry_id,
                        existing_entry,
                    )
                    drop_ids.append(entry_id)
                    continue
                self._entry_ids[event.event_id] = entry_id
                self._held_entries.add(entry_id)
                self._in_flight.add(event.event_id)
            events.append(event)
        if drop_ids:
            try:
                client.xack(self._stream, self._group, *drop_ids)
            except Exception:
                logger.exception("Failed to XACK discarded Redis Streams entries")
        return events
