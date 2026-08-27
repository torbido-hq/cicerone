"""S3-compatible EventSource: R2/MinIO list+marker (primary); AWS SQS optional."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import boto3
from botocore.config import Config

from cicerone.config import ConfigError
from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.normalize import EventNormalizeError, normalize_event
from cicerone.io.options import build_s3_client, require_option

logger = logging.getLogger(__name__)

_MODES = frozenset({"sqs", "list"})
_DEFAULT_LIST_PAGE_SIZE = 100
_DEFAULT_SQS_LAG_CACHE_TTL_SECONDS = 5.0
_DEFAULT_SQS_CLIENT_TIMEOUT_SECONDS = 2.0
# Transient S3 read errors retry; after this many failures the object is skipped.
_LOAD_FAILURE_SKIP_AFTER = 3
# Cover lock-busy nack retries so the message is not stolen mid-lease wait.
_SQS_NACK_VISIBILITY_TIMEOUT_SECONDS = 60
# In-flight apply (online fit_partial) can outlast the receive visibility window.
_SQS_APPLY_VISIBILITY_TIMEOUT_SECONDS = 300


def validate_s3_event_options(options: dict[str, Any]) -> str:
    """Validate ``events.options`` for ``kind=s3``; return resolved mode."""
    for key in ("access_key_id", "secret_access_key", "bucket"):
        if not options.get(key):
            raise ConfigError(f'events.options.{key} is required when events.kind = "s3"')
    mode = options.get("mode")
    if mode is None:
        resolved = "sqs" if options.get("queue_url") else "list"
    else:
        resolved = str(mode).lower()
        if resolved not in _MODES:
            raise ConfigError(f"events.options.mode must be one of {sorted(_MODES)}, got {mode!r}")
    if resolved == "sqs":
        if options.get("endpoint_url"):
            raise ConfigError(
                'events.options.mode = "sqs" is AWS-only; '
                'S3-compatible endpoints (R2/MinIO) must use mode = "list"'
            )
        if not options.get("queue_url"):
            raise ConfigError('events.options.queue_url is required when events.options.mode = "sqs"')
    return resolved


def _as_int(options: dict[str, Any], key: str, default: int) -> int:
    raw = options.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"events.options.{key} must be an integer, got {raw!r}") from exc


def _positive_int(options: dict[str, Any], key: str, default: int) -> int:
    value = _as_int(options, key, default)
    if value < 1:
        raise ConfigError(f"events.options.{key} must be an integer >= 1, got {value}")
    return value


def _positive_float(options: dict[str, Any], key: str, default: float) -> float:
    raw = options.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"events.options.{key} must be a float > 0, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"events.options.{key} must be > 0, got {value}")
    return value


# One SQS message / S3 object → _Batch. Events go pending → (on poll) in_flight;
# ack clears remaining; marker/SQS delete only for leading fully-acked batches.
@dataclass
class _Batch:
    remaining: set[str]
    receipt_handle: str | None = None
    object_key: str | None = None
    object_etag: str | None = None
    event_ids: set[str] = field(default_factory=set)


def _optional_aws_region(options: dict[str, Any]) -> str | None:
    if options.get("region_name"):
        return str(options["region_name"])
    return None


def _normalize_prefix(raw: Any) -> str:
    prefix = str(raw or "").strip("/")
    return f"{prefix}/" if prefix else ""


def _stable_event_id(payload: dict[str, Any], *, bucket: str, key: str, etag: str, index: int) -> str:
    existing = payload.get("event_id") or payload.get("idempotency_key")
    if existing not in (None, ""):
        return str(existing)
    return f"{bucket}/{key}|{etag}|{index}"


def _events_from_body(body: bytes, *, bucket: str, key: str, etag: str) -> list[NormalizedEvent]:
    text = body.decode("utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in s3://{bucket}/{key}") from exc
    if isinstance(data, dict):
        payloads: list[Any] = [data]
    elif isinstance(data, list):
        payloads = data
    else:
        raise ValueError(f"s3://{bucket}/{key} must be a JSON object or array")
    events: list[NormalizedEvent] = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            logger.warning(
                "Skipping non-object event %d in s3://%s/%s",
                index,
                bucket,
                key,
            )
            continue
        try:
            event_id = _stable_event_id(payload, bucket=bucket, key=key, etag=etag, index=index)
            events.append(normalize_event({**payload, "event_id": event_id}))
        except EventNormalizeError:
            logger.warning(
                "Skipping invalid event %d in s3://%s/%s",
                index,
                bucket,
                key,
                exc_info=True,
            )
    return events


def _s3_records_from_sqs_body(body: str) -> list[tuple[str, str]]:
    data = json.loads(body)
    if isinstance(data, dict) and "Message" in data and ("TopicArn" in data or "Type" in data):
        message = data["Message"]
        data = json.loads(message) if isinstance(message, str) else message
    if not isinstance(data, dict):
        raise ValueError("SQS message body must be a JSON object")
    records = data.get("Records")
    if not isinstance(records, list):
        raise ValueError("SQS message missing S3 Records")
    out: list[tuple[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        event_name = str(record.get("eventName") or "")
        if event_name and not event_name.startswith("ObjectCreated"):
            continue
        s3 = record.get("s3") or {}
        if not isinstance(s3, dict):
            continue
        bucket = (s3.get("bucket") or {}).get("name")
        key = (s3.get("object") or {}).get("key")
        if bucket and key:
            out.append((str(bucket), unquote_plus(str(key))))
    return out


class S3EventSource(EventSource):
    """R2/MinIO list+marker poll, or AWS SQS notifications (no endpoint_url)."""

    def __init__(self, options: dict[str, Any] | None = None):
        options = dict(options or {})
        self._options = options
        self._mode = validate_s3_event_options(options)
        self._bucket = str(options["bucket"])
        self._prefix = _normalize_prefix(options.get("prefix"))
        self._queue_url = str(options["queue_url"]) if self._mode == "sqs" else None
        self._marker_path = Path(options["marker_path"]) if options.get("marker_path") else None
        self._marker_key = str(options.get("initial_marker") or "")
        self._wait_time_seconds = max(0, _as_int(options, "wait_time_seconds", 0))
        self._max_messages = max(1, min(10, _as_int(options, "max_messages", 10)))
        self._list_page_size = _positive_int(options, "list_page_size", _DEFAULT_LIST_PAGE_SIZE)
        self._sqs_lag_cache_ttl_seconds = _positive_float(
            options, "sqs_lag_cache_ttl_seconds", _DEFAULT_SQS_LAG_CACHE_TTL_SECONDS
        )
        self._sqs_client_timeout_seconds = _positive_float(
            options, "sqs_client_timeout_seconds", _DEFAULT_SQS_CLIENT_TIMEOUT_SECONDS
        )
        self._s3 = None
        self._sqs = None
        self._lock = threading.Lock()
        self._connected = False
        self._last_event_at: datetime | None = None
        self._pending: deque[NormalizedEvent] = deque()
        self._in_flight: OrderedDict[str, NormalizedEvent] = OrderedDict()
        self._event_batch: dict[str, _Batch] = {}
        self._batches: OrderedDict[str, _Batch] = OrderedDict()
        self._batch_seq = 0
        self._load_failures: dict[str, int] = {}
        self._sqs_visible_lag: int | None = None
        self._sqs_visible_lag_at: float = 0.0

    def connect(self) -> None:
        with self._lock:
            if self._s3 is None:
                self._s3 = build_s3_client(self._options)
            if self._mode == "sqs" and self._sqs is None:
                sqs_kwargs: dict[str, Any] = {
                    "aws_access_key_id": require_option(self._options, "access_key_id", "s3"),
                    "aws_secret_access_key": require_option(self._options, "secret_access_key", "s3"),
                    "config": Config(
                        retries={"max_attempts": 3, "mode": "standard"},
                        connect_timeout=self._sqs_client_timeout_seconds,
                        read_timeout=self._sqs_client_timeout_seconds,
                    ),
                }
                region_name = _optional_aws_region(self._options)
                if region_name is not None:
                    sqs_kwargs["region_name"] = region_name
                self._sqs = boto3.client("sqs", **sqs_kwargs)
            if self._mode == "list":
                self._load_marker_unlocked()
            self._connected = True

    def close(self) -> None:
        with self._lock:
            self._s3 = None
            self._sqs = None
            self._connected = False

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        with self._lock:
            if self._s3 is None:
                raise RuntimeError("S3EventSource.connect() required before poll")
        out = self._drain_pending(max_events)
        if len(out) >= max_events:
            return out
        if self._mode == "sqs":
            self._fetch_sqs(max_events - len(out))
        else:
            self._fetch_list(max_events - len(out))
        out.extend(self._drain_pending(max_events - len(out)))
        return out

    def nack(self, events: Sequence[NormalizedEvent]) -> None:
        receipts: list[str] = []
        with self._lock:
            pending_ids = {item.event_id for item in self._pending}
            seen_batches: set[int] = set()
            for event in reversed(list(events)):
                eid = event.event_id
                self._in_flight.pop(eid, None)
                batch = self._event_batch.get(eid)
                if batch is None:
                    continue
                if eid not in pending_ids:
                    self._pending.appendleft(event)
                    pending_ids.add(eid)
                if batch.receipt_handle and id(batch) not in seen_batches:
                    seen_batches.add(id(batch))
                    receipts.append(batch.receipt_handle)
            sqs = self._sqs
            queue_url = self._queue_url
        self._extend_sqs_visibility(
            receipts,
            sqs=sqs,
            queue_url=queue_url,
            timeout_seconds=_SQS_NACK_VISIBILITY_TIMEOUT_SECONDS,
        )

    def heartbeat(self, events: Sequence[NormalizedEvent]) -> None:
        receipts: list[str] = []
        with self._lock:
            seen_batches: set[int] = set()
            for event in events:
                batch = self._event_batch.get(event.event_id)
                if batch is None or not batch.receipt_handle:
                    continue
                if id(batch) in seen_batches:
                    continue
                seen_batches.add(id(batch))
                receipts.append(batch.receipt_handle)
            sqs = self._sqs
            queue_url = self._queue_url
        self._extend_sqs_visibility(
            receipts,
            sqs=sqs,
            queue_url=queue_url,
            timeout_seconds=_SQS_APPLY_VISIBILITY_TIMEOUT_SECONDS,
        )

    def ack(self, event_ids: Sequence[str]) -> None:
        completed: list[_Batch] = []
        with self._lock:
            for event_id in event_ids:
                eid = str(event_id)
                self._in_flight.pop(eid, None)
                batch = self._event_batch.pop(eid, None)
                if batch is None:
                    continue
                batch.remaining.discard(eid)
            completed.extend(self._pop_completed_batches_unlocked())
            sqs = self._sqs
            queue_url = self._queue_url
        self._finish_completed(completed, sqs=sqs, queue_url=queue_url)

    def health(self) -> EventSourceHealth:
        with self._lock:
            held = len(self._pending) + len(self._in_flight)
            if self._mode == "sqs":
                sqs = self._sqs
                queue_url = self._queue_url
                cached_visible = self._sqs_visible_lag
                cached_at = self._sqs_visible_lag_at
            else:
                sqs = None
                queue_url = None
                cached_visible = None
                cached_at = 0.0
            connected = self._connected
            last_event_at = self._last_event_at
            detail = f"s3 mode={self._mode} bucket={self._bucket}"
            if self._prefix:
                detail = f"{detail} prefix={self._prefix}"
        lag = held
        if sqs is not None and queue_url:
            now = time.monotonic()
            visible: int | None = None
            if cached_visible is not None and (now - cached_at) < self._sqs_lag_cache_ttl_seconds:
                visible = cached_visible
            else:
                try:
                    attrs = sqs.get_queue_attributes(
                        QueueUrl=queue_url,
                        AttributeNames=["ApproximateNumberOfMessages"],
                    ).get("Attributes", {})
                    visible = int(attrs.get("ApproximateNumberOfMessages", 0))
                    with self._lock:
                        self._sqs_visible_lag = visible
                        self._sqs_visible_lag_at = now
                except Exception:
                    logger.exception("Failed to estimate SQS event lag")
                    visible = cached_visible
            if visible is not None:
                # Visible queue depth + local held work (not_visible is already in held
                # once received/parsed; avoid double-counting it).
                lag = held + visible
        return EventSourceHealth(
            connected=connected,
            lag=lag,
            last_event_at=last_event_at,
            detail=detail,
        )

    def _drain_pending(self, limit: int) -> list[NormalizedEvent]:
        out: list[NormalizedEvent] = []
        with self._lock:
            while self._pending and len(out) < limit:
                event = self._pending.popleft()
                self._in_flight[event.event_id] = event
                out.append(event)
        return out

    def _fetch_list(self, need: int) -> None:
        if need < 1:
            return
        with self._lock:
            s3 = self._s3
            bucket = self._bucket
            prefix = self._prefix
            marker = self._marker_key
            page_size = self._list_page_size
            held_ids = set(self._in_flight) | {event.event_id for event in self._pending}
            held_keys = {batch.object_key for batch in self._batches.values() if batch.object_key}
            if s3 is None:
                raise RuntimeError("S3EventSource.connect() required before poll")
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "MaxKeys": min(page_size, max(need, 1)),
        }
        if prefix:
            kwargs["Prefix"] = prefix
        if marker:
            kwargs["StartAfter"] = marker
        page = s3.list_objects_v2(**kwargs)
        loaded = 0
        for obj in page.get("Contents") or []:
            if loaded >= need:
                break
            key = str(obj["Key"])
            if key in held_keys:
                continue
            etag = str(obj.get("ETag") or "")
            try:
                events = self._load_object_events(s3, bucket, key, etag=etag)
            except ValueError:
                logger.exception("Skipping unreadable s3://%s/%s", bucket, key)
                self._register_batch([], object_key=key, object_etag=etag)
                self._load_failures.pop(key, None)
                continue
            except Exception:
                failures = self._load_failures.get(key, 0) + 1
                self._load_failures[key] = failures
                logger.exception(
                    "Failed to read s3://%s/%s (%d/%d)",
                    bucket,
                    key,
                    failures,
                    _LOAD_FAILURE_SKIP_AFTER,
                )
                if failures >= _LOAD_FAILURE_SKIP_AFTER:
                    logger.error(
                        "Skipping unreadable s3://%s/%s after %d load failures",
                        bucket,
                        key,
                        failures,
                    )
                    self._register_batch([], object_key=key, object_etag=etag)
                    self._load_failures.pop(key, None)
                    continue
                break
            self._load_failures.pop(key, None)
            novel = [event for event in events if event.event_id not in held_ids]
            if not novel:
                self._register_batch([], object_key=key, object_etag=etag)
                continue
            self._register_batch(novel, object_key=key, object_etag=etag)
            held_ids.update(event.event_id for event in novel)
            held_keys.add(key)
            loaded += len(novel)

    def _matching_sqs_records(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        matched: list[tuple[str, str]] = []
        for bucket, key in pairs:
            if bucket != self._bucket:
                continue
            if self._prefix and not key.startswith(self._prefix):
                continue
            matched.append((bucket, key))
        return matched

    def _fetch_sqs(self, need: int) -> None:
        if need < 1:
            return
        with self._lock:
            sqs = self._sqs
            s3 = self._s3
            queue_url = self._queue_url
            held_ids = set(self._in_flight) | {event.event_id for event in self._pending}
            if s3 is None or sqs is None or queue_url is None:
                raise RuntimeError("S3EventSource.connect() required before poll")
        loaded = 0
        while loaded < need:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=min(self._max_messages, 10),
                WaitTimeSeconds=self._wait_time_seconds if loaded == 0 else 0,
            )
            messages = response.get("Messages") or []
            if not messages:
                break
            made_progress = False
            for message in messages:
                if loaded >= need:
                    break
                receipt = message["ReceiptHandle"]
                try:
                    pairs = _s3_records_from_sqs_body(message["Body"])
                except Exception:
                    logger.exception("Invalid S3 notification on SQS; deleting poison message")
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    made_progress = True
                    continue
                if not pairs:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    made_progress = True
                    continue
                matched = self._matching_sqs_records(pairs)
                if not matched:
                    # Shared queue / miswired notification — leave message for others.
                    logger.warning(
                        "Ignoring SQS S3 notification with no keys under bucket=%s prefix=%r",
                        self._bucket,
                        self._prefix,
                    )
                    continue
                batch_events: list[NormalizedEvent] = []
                failed = False
                for bucket, key in matched:
                    try:
                        batch_events.extend(self._load_object_events(s3, bucket, key))
                    except Exception:
                        logger.exception(
                            "Failed to load s3://%s/%s from SQS notification; leaving message for retry",
                            bucket,
                            key,
                        )
                        failed = True
                        break
                if failed:
                    continue
                novel = [event for event in batch_events if event.event_id not in held_ids]
                if not novel:
                    # Already holding these events (local nack retry). Keep the
                    # latest receipt so ack can still delete after visibility refresh.
                    self._adopt_sqs_receipt(receipt, {event.event_id for event in batch_events})
                    made_progress = True
                    continue
                self._register_batch(novel, receipt_handle=receipt)
                held_ids.update(event.event_id for event in novel)
                loaded += len(novel)
                made_progress = True
            if not made_progress:
                break

    def _load_object_events(self, s3, bucket: str, key: str, etag: str = "") -> list[NormalizedEvent]:
        obj = s3.get_object(Bucket=bucket, Key=key)
        resolved_etag = etag or str(obj.get("ETag") or "")
        return _events_from_body(
            obj["Body"].read(),
            bucket=bucket,
            key=key,
            etag=resolved_etag,
        )

    def _register_batch(
        self,
        events: list[NormalizedEvent],
        *,
        receipt_handle: str | None = None,
        object_key: str | None = None,
        object_etag: str | None = None,
    ) -> None:
        # pending holds undelivered events; poll moves them to in_flight.
        # _batches stays ordered: marker/SQS delete only for a leading batch
        # whose remaining is empty (empty skip batches complete immediately).
        event_ids = {event.event_id for event in events}
        batch = _Batch(
            remaining=set(event_ids),
            receipt_handle=receipt_handle,
            object_key=object_key,
            object_etag=object_etag,
            event_ids=set(event_ids),
        )
        completed: list[_Batch] = []
        with self._lock:
            self._batch_seq += 1
            batch_key = f"{self._batch_seq}:{object_key or receipt_handle or self._batch_seq}"
            self._batches[batch_key] = batch
            for event in events:
                self._event_batch[event.event_id] = batch
                self._pending.append(event)
                self._last_event_at = event.occurred_at
            # Empty / skip batches complete immediately when they lead the queue.
            completed.extend(self._pop_completed_batches_unlocked())
            sqs = self._sqs
            queue_url = self._queue_url
        self._finish_completed(completed, sqs=sqs, queue_url=queue_url)

    def _pop_completed_batches_unlocked(self) -> list[_Batch]:
        completed: list[_Batch] = []
        while self._batches:
            _key, batch = next(iter(self._batches.items()))
            if batch.remaining:
                break
            self._batches.popitem(last=False)
            completed.append(batch)
            if batch.object_key is not None:
                self._marker_key = batch.object_key
        if completed and self._mode == "list":
            self._persist_marker_unlocked()
        return completed

    def _finish_completed(
        self,
        completed: list[_Batch],
        *,
        sqs: Any,
        queue_url: str | None,
    ) -> None:
        for batch in completed:
            if batch.receipt_handle and sqs is not None and queue_url:
                try:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=batch.receipt_handle)
                except Exception:
                    logger.exception("Failed to delete SQS message after ack")

    def _extend_sqs_visibility(
        self,
        receipts: Sequence[str],
        *,
        sqs: Any,
        queue_url: str | None,
        timeout_seconds: int = _SQS_NACK_VISIBILITY_TIMEOUT_SECONDS,
    ) -> None:
        if not receipts or sqs is None or queue_url is None:
            return
        for receipt in receipts:
            try:
                sqs.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt,
                    VisibilityTimeout=timeout_seconds,
                )
            except Exception:
                logger.exception("Failed to extend SQS visibility")

    def _adopt_sqs_receipt(self, receipt: str, event_ids: set[str]) -> None:
        with self._lock:
            for eid in event_ids:
                batch = self._event_batch.get(eid)
                if batch is not None:
                    batch.receipt_handle = receipt
                    return

    def _load_marker_unlocked(self) -> None:
        if self._marker_path is None or not self._marker_path.is_file():
            return
        try:
            raw = json.loads(self._marker_path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("marker root must be an object")
            key = raw.get("key")
            if key:
                self._marker_key = str(key)
        except Exception:
            logger.exception(
                "Ignoring corrupt marker file %s; keeping marker %r",
                self._marker_path,
                self._marker_key,
            )

    def _persist_marker_unlocked(self) -> None:
        if self._marker_path is None:
            return
        self._marker_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": self._marker_key}
        tmp = self._marker_path.with_name(f".{self._marker_path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self._marker_path)
        try:
            dir_fd = os.open(str(self._marker_path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
