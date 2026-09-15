"""SQS notification poll methods for S3EventSource."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from typing import Any

from cicerone.events.base import NormalizedEvent
from cicerone.events.s3_parse import (
    _SQS_APPLY_VISIBILITY_TIMEOUT_SECONDS,
    _SQS_NACK_VISIBILITY_TIMEOUT_SECONDS,
    _Batch,
    _s3_records_from_sqs_body,
)

logger = logging.getLogger("cicerone.events.s3")


class S3SqsPoll:
    _lock: threading.Lock
    _s3: Any
    _sqs: Any
    _queue_url: str | None
    _bucket: str
    _prefix: str
    _max_messages: int
    _wait_time_seconds: int
    _in_flight: OrderedDict[str, NormalizedEvent]
    _pending: deque[NormalizedEvent]
    _event_batch: dict[str, _Batch]
    _load_object_events: Callable[..., list[NormalizedEvent]]
    _register_batch: Callable[..., None]

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
                VisibilityTimeout=_SQS_APPLY_VISIBILITY_TIMEOUT_SECONDS,
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
