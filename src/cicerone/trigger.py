"""Retrain webhook + optional input poll; shares RunGuard with cron."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from cicerone import job
from cicerone.config import IOSettings, Settings
from cicerone.http_auth import optional_bearer_deps
from cicerone.http_security import SecurityHeadersMiddleware
from cicerone.io.options import build_s3_client, object_key, validate_storage_options
from cicerone.locks import LockBackend
from cicerone.locks.redis import RedisError
from cicerone.serve.metrics import record_retrain_trigger

logger = logging.getLogger(__name__)

_LOCK_BACKEND_ERRORS = (SQLAlchemyError, RedisError)


class _RunFn(Protocol):
    def __call__(self, triggered_by: str, *, fence_check: Callable[[], bool] | None = None) -> None: ...


class RunGuard:
    """At most one run at a time; triggers within debounce_seconds are ignored."""

    def __init__(
        self,
        debounce_seconds: float,
        run_fn: _RunFn = job.run,
        lock_backend: LockBackend | None = None,
    ):
        self._debounce_seconds = debounce_seconds
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self._backend = lock_backend
        self._running = False
        self._last_started_at: float | None = None
        self._last_error: BaseException | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running

    @property
    def last_error(self) -> BaseException | None:
        with self._lock:
            return self._last_error

    def trigger(self, triggered_by: str) -> bool:
        with self._lock:
            now = time.monotonic()
            if self._running:
                logger.info("Ignoring %s trigger: a run is already in progress", triggered_by)
                record_retrain_trigger(triggered_by, accepted=False)
                return False
            if self._last_started_at is not None and now - self._last_started_at < self._debounce_seconds:
                logger.info("Ignoring %s trigger: within debounce window", triggered_by)
                record_retrain_trigger(triggered_by, accepted=False)
                return False
            if self._backend is not None and not self._backend.acquire():
                logger.info("Ignoring %s trigger: distributed lock held by another instance", triggered_by)
                record_retrain_trigger(triggered_by, accepted=False)
                return False
            self._running = True
            self._last_started_at = now
            self._last_error = None

        record_retrain_trigger(triggered_by, accepted=True)
        threading.Thread(target=self._run, args=(triggered_by,), daemon=True).start()
        return True

    def _run(self, triggered_by: str) -> None:
        try:
            if self._backend is None:
                self._run_fn(triggered_by=triggered_by)
            else:
                self._run_fn(triggered_by=triggered_by, fence_check=self._backend.owned)
        except Exception as exc:
            logger.exception("Triggered run (%s) failed", triggered_by)
            with self._lock:
                self._last_error = exc
        finally:
            try:
                if self._backend is not None:
                    self._backend.release()
            finally:
                with self._lock:
                    self._running = False


def _current_marker(input_settings: IOSettings) -> str | None:
    options = input_settings.options
    try:
        backend = validate_storage_options(options)
        if backend == "local":
            path = Path(options["path"]) / "events.parquet"
            if not path.exists():
                return None
            return str(path.stat().st_mtime)

        client = build_s3_client(options)
        bucket = options["bucket"]
        key = object_key(options, "events.parquet")
        head = client.head_object(Bucket=bucket, Key=key)
        return str(head["LastModified"])
    except Exception:
        logger.exception("Failed to check input source for changes; treating as unchanged")
        return None


def poll_input_forever(input_settings: IOSettings, guard: RunGuard, interval_seconds: float) -> None:
    if input_settings.kind != "dataset":
        logger.warning(
            "trigger.poll_input_bucket is enabled but input.kind is %r; polling only "
            "supports 'dataset' inputs, disabling the poller",
            input_settings.kind,
        )
        return

    last_marker = _current_marker(input_settings)
    while True:
        time.sleep(interval_seconds)
        marker = _current_marker(input_settings)
        if marker is None or marker == last_marker:
            continue
        try:
            accepted = guard.trigger("s3-poll")
        except _LOCK_BACKEND_ERRORS:
            logger.warning("Input poll trigger failed on the lock backend; retrying next poll", exc_info=True)
            continue
        # A rejected trigger (busy, debounced, lock outage) keeps the change pending for the next poll.
        if accepted:
            last_marker = marker


def create_app(settings: Settings, guard: RunGuard) -> FastAPI:
    app = FastAPI(
        title="cicerone-trigger",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    dependencies = optional_bearer_deps(settings.trigger.auth_token)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/trigger/retrain", dependencies=dependencies, status_code=202)
    def trigger_retrain() -> dict[str, Any]:
        started = guard.trigger("webhook")
        if not started:
            raise HTTPException(status_code=429, detail="A run is already in progress or was just triggered")
        return {"status": "started"}

    return app
