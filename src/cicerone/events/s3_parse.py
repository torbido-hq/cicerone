"""S3 event option validation and payload parse helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote_plus

from cicerone.config import ConfigError
from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import EventNormalizeError, normalize_event

logger = logging.getLogger("cicerone.events.s3")

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
