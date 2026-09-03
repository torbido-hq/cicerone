"""S3-compatible EventSource: R2/MinIO list+marker (primary); AWS SQS optional."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.s3_list import S3ListPoll
from cicerone.events.s3_parse import (
    _DEFAULT_LIST_PAGE_SIZE,
    _DEFAULT_SQS_CLIENT_TIMEOUT_SECONDS,
    _DEFAULT_SQS_LAG_CACHE_TTL_SECONDS,
    _SQS_APPLY_VISIBILITY_TIMEOUT_SECONDS,
    _SQS_NACK_VISIBILITY_TIMEOUT_SECONDS,
    _as_int,
    _Batch,
    _events_from_body,
    _normalize_prefix,
    _optional_aws_region,
    _positive_float,
    _positive_int,
    validate_s3_event_options,
)
from cicerone.events.s3_parse import _LOAD_FAILURE_SKIP_AFTER as _LOAD_FAILURE_SKIP_AFTER
from cicerone.events.s3_parse import _s3_records_from_sqs_body as _s3_records_from_sqs_body
from cicerone.events.s3_sqs import S3SqsPoll
from cicerone.io.options import build_s3_client, require_option

logger = logging.getLogger(__name__)


class S3EventSource(S3ListPoll, S3SqsPoll, EventSource):
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
