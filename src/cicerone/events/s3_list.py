"""List+marker poll methods for S3EventSource."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cicerone.events.base import NormalizedEvent
from cicerone.events.s3_parse import _LOAD_FAILURE_SKIP_AFTER, _Batch

logger = logging.getLogger("cicerone.events.s3")


class S3ListPoll:
    _lock: threading.Lock
    _s3: Any
    _bucket: str
    _prefix: str
    _marker_key: str
    _marker_path: Path | None
    _list_page_size: int
    _in_flight: OrderedDict[str, NormalizedEvent]
    _pending: deque[NormalizedEvent]
    _batches: OrderedDict[str, _Batch]
    _load_failures: dict[str, int]
    _load_object_events: Callable[..., list[NormalizedEvent]]
    _register_batch: Callable[..., None]

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
