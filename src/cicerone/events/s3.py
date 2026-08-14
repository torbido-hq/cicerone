"""S3 EventSource: SQS notifications (AWS) or list/marker poll (R2/MinIO)."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from cicerone.config import ConfigError
from cicerone.events.base import EventSourceHealth, NormalizedEvent
from cicerone.events.normalize import EventNormalizeError, normalize_event
from cicerone.io.options import require_option

logger = logging.getLogger(__name__)

_LAG_LIST_LIMIT = 1_000
_MODES = frozenset({"sqs", "list"})


@dataclass
class _Batch:
    remaining: set[str]
    receipt_handle: str | None = None
    object_key: str | None = None
    object_etag: str | None = None
    event_ids: set[str] = field(default_factory=set)


def _region_name(options: dict[str, Any]) -> str:
    if options.get("region_name"):
        return str(options["region_name"])
    # R2/MinIO-style endpoints expect "auto"; AWS SQS/S3 need a real region.
    return "auto" if options.get("endpoint_url") else "us-east-1"


def _boto_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint_url": options.get("endpoint_url"),
        "aws_access_key_id": require_option(options, "access_key_id", "s3"),
        "aws_secret_access_key": require_option(options, "secret_access_key", "s3"),
        "region_name": _region_name(options),
        "config": Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    }


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
            raise ValueError(f"s3://{bucket}/{key} event {index} must be a JSON object")
        event_id = _stable_event_id(payload, bucket=bucket, key=key, etag=etag, index=index)
        events.append(normalize_event({**payload, "event_id": event_id}))
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


class S3EventSource:
    """Poll interaction JSON objects from S3 via SQS notifications or list/marker."""

    def __init__(self, options: dict[str, Any] | None = None):
        options = dict(options or {})
        self._options = options
        self._bucket = str(require_option(options, "bucket", "s3"))
        self._prefix = str(options.get("prefix") or "").lstrip("/")
        mode = options.get("mode")
        if mode is None:
            mode = "sqs" if options.get("queue_url") else "list"
        mode = str(mode).lower()
        if mode not in _MODES:
            raise ConfigError(f"events.options.mode must be one of {sorted(_MODES)}, got {mode!r}")
        self._mode = mode
        self._queue_url = options.get("queue_url")
        if self._mode == "sqs":
            if not self._queue_url:
                raise ConfigError('events.options.queue_url is required when events.options.mode = "sqs"')
            self._queue_url = str(self._queue_url)
        self._marker_path = Path(options["marker_path"]) if options.get("marker_path") else None
        self._marker_key = str(options.get("initial_marker") or "")
        self._wait_time_seconds = max(0, int(options.get("wait_time_seconds", 0)))
        self._max_messages = max(1, min(10, int(options.get("max_messages", 10))))
        self._s3 = None
        self._sqs = None
        self._lock = threading.Lock()
        self._connected = False
        self._last_event_at: datetime | None = None
        self._in_flight: OrderedDict[str, NormalizedEvent] = OrderedDict()
        self._event_batch: dict[str, _Batch] = {}
        self._batches: OrderedDict[str, _Batch] = OrderedDict()
        self._batch_seq = 0

    def connect(self) -> None:
        with self._lock:
            kwargs = _boto_kwargs(self._options)
            if self._s3 is None:
                self._s3 = boto3.client("s3", **kwargs)
            if self._mode == "sqs" and self._sqs is None:
                self._sqs = boto3.client("sqs", **kwargs)
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
            mode = self._mode
        if mode == "sqs":
            return self._poll_sqs(max_events)
        return self._poll_list(max_events)

    def nack(self, events: Sequence[NormalizedEvent]) -> None:
        with self._lock:
            for event in events:
                self._drop_in_flight_unlocked(event.event_id)

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
            sqs = self._sqs
            queue_url = self._queue_url
        for batch in completed:
            if batch.receipt_handle and sqs is not None and queue_url:
                try:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=batch.receipt_handle)
                except ClientError:
                    logger.exception("Failed to delete SQS message after ack")

    def health(self) -> EventSourceHealth:
        with self._lock:
            connected = self._connected
            last_event_at = self._last_event_at
            detail = f"s3 mode={self._mode} bucket={self._bucket}"
            in_flight = len(self._in_flight)
            mode = self._mode
            marker = self._marker_key
            s3 = self._s3
            sqs = self._sqs
            queue_url = self._queue_url
            prefix = self._prefix
            bucket = self._bucket
        lag: int | None = in_flight
        try:
            if mode == "sqs" and sqs is not None and queue_url:
                attrs = sqs.get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=[
                        "ApproximateNumberOfMessages",
                        "ApproximateNumberOfMessagesNotVisible",
                    ],
                ).get("Attributes", {})
                visible = int(attrs.get("ApproximateNumberOfMessages", 0))
                not_visible = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
                lag = visible + not_visible
            elif mode == "list" and s3 is not None:
                lag = self._count_list_lag(s3, bucket, prefix, marker, in_flight)
        except Exception:
            logger.exception("Failed to estimate S3 event lag")
        return EventSourceHealth(
            connected=connected,
            lag=lag,
            last_event_at=last_event_at,
            detail=detail,
        )

    def _count_list_lag(self, s3, bucket: str, prefix: str, marker: str, in_flight: int) -> int | None:
        kwargs: dict[str, Any] = {"Bucket": bucket, "MaxKeys": _LAG_LIST_LIMIT}
        if prefix:
            kwargs["Prefix"] = prefix
        if marker:
            kwargs["StartAfter"] = marker
        listed = 0
        while listed < _LAG_LIST_LIMIT:
            page = s3.list_objects_v2(**kwargs)
            contents = page.get("Contents") or []
            listed += len(contents)
            if not page.get("IsTruncated"):
                return listed + in_flight
            kwargs["StartAfter"] = contents[-1]["Key"]
            if listed >= _LAG_LIST_LIMIT:
                break
        return None

    def _poll_sqs(self, max_events: int) -> list[NormalizedEvent]:
        with self._lock:
            sqs = self._sqs
            s3 = self._s3
            queue_url = self._queue_url
            in_flight_ids = set(self._in_flight)
        assert sqs is not None and s3 is not None and queue_url is not None
        out: list[NormalizedEvent] = []
        while len(out) < max_events:
            remaining = max_events - len(out)
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=min(self._max_messages, max(1, remaining)),
                WaitTimeSeconds=self._wait_time_seconds if not out else 0,
                MessageAttributeNames=["All"],
            )
            messages = response.get("Messages") or []
            if not messages:
                break
            for message in messages:
                receipt = message["ReceiptHandle"]
                try:
                    pairs = _s3_records_from_sqs_body(message["Body"])
                except Exception:
                    logger.exception("Invalid S3 notification on SQS; deleting poison message")
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    continue
                batch_events: list[NormalizedEvent] = []
                for bucket, key in pairs:
                    try:
                        batch_events.extend(self._load_object_events(s3, bucket, key))
                    except Exception:
                        logger.exception("Failed to load s3://%s/%s from SQS notification", bucket, key)
                novel = [event for event in batch_events if event.event_id not in in_flight_ids]
                if not novel:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    continue
                self._register_batch(novel, receipt_handle=receipt)
                in_flight_ids.update(event.event_id for event in novel)
                out.extend(novel)
                if len(out) >= max_events:
                    break
        return out

    def _poll_list(self, max_events: int) -> list[NormalizedEvent]:
        with self._lock:
            s3 = self._s3
            bucket = self._bucket
            prefix = self._prefix
            marker = self._marker_key
            in_flight_ids = set(self._in_flight)
            in_flight_keys = {batch.object_key for batch in self._batches.values() if batch.object_key}
        assert s3 is not None
        kwargs: dict[str, Any] = {"Bucket": bucket, "MaxKeys": max(1, max_events)}
        if prefix:
            kwargs["Prefix"] = prefix
        if marker:
            kwargs["StartAfter"] = marker
        page = s3.list_objects_v2(**kwargs)
        out: list[NormalizedEvent] = []
        for obj in page.get("Contents") or []:
            if len(out) >= max_events:
                break
            key = str(obj["Key"])
            if key in in_flight_keys:
                continue
            try:
                events = self._load_object_events(s3, bucket, key, etag=str(obj.get("ETag") or ""))
            except Exception:
                logger.exception("Skipping unreadable s3://%s/%s", bucket, key)
                self._register_batch([], object_key=key, object_etag=str(obj.get("ETag") or ""))
                continue
            novel = [event for event in events if event.event_id not in in_flight_ids]
            if not novel:
                self._register_batch([], object_key=key, object_etag=str(obj.get("ETag") or ""))
                continue
            self._register_batch(
                novel,
                object_key=key,
                object_etag=str(obj.get("ETag") or ""),
            )
            in_flight_ids.update(event.event_id for event in novel)
            out.extend(novel)
        if not out:
            # Advance marker past empty/skipped leading objects already registered.
            self.ack([])
        return out

    def _load_object_events(self, s3, bucket: str, key: str, etag: str = "") -> list[NormalizedEvent]:
        obj = s3.get_object(Bucket=bucket, Key=key)
        resolved_etag = etag or str(obj.get("ETag") or "")
        body = obj["Body"].read()
        try:
            return _events_from_body(body, bucket=bucket, key=key, etag=resolved_etag)
        except (EventNormalizeError, ValueError):
            logger.exception("Invalid event payload in s3://%s/%s", bucket, key)
            raise

    def _register_batch(
        self,
        events: list[NormalizedEvent],
        *,
        receipt_handle: str | None = None,
        object_key: str | None = None,
        object_etag: str | None = None,
    ) -> None:
        event_ids = {event.event_id for event in events}
        batch = _Batch(
            remaining=set(event_ids),
            receipt_handle=receipt_handle,
            object_key=object_key,
            object_etag=object_etag,
            event_ids=event_ids,
        )
        with self._lock:
            self._batch_seq += 1
            batch_key = f"{self._batch_seq}:{object_key or receipt_handle or self._batch_seq}"
            self._batches[batch_key] = batch
            for event in events:
                self._in_flight[event.event_id] = event
                self._event_batch[event.event_id] = batch
                self._last_event_at = event.occurred_at

    def _drop_in_flight_unlocked(self, event_id: str) -> None:
        eid = str(event_id)
        self._in_flight.pop(eid, None)
        batch = self._event_batch.pop(eid, None)
        if batch is None:
            return
        batch.remaining.discard(eid)
        batch.event_ids.discard(eid)
        # Drop the whole SQS/list batch so visibility timeout / re-list can retry.
        for key, candidate in list(self._batches.items()):
            if candidate is batch:
                self._batches.pop(key, None)
                break
        for other_id in list(batch.event_ids):
            self._in_flight.pop(other_id, None)
            self._event_batch.pop(other_id, None)

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
        try:
            dir_fd = os.open(str(tmp.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        tmp.replace(self._marker_path)
