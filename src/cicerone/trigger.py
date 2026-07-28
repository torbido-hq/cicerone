"""Event-driven retraining trigger: runs alongside the existing cron
schedule in cicerone.scheduler (additive, cron keeps working unchanged).
Two ways to signal a run:

  - POST /trigger/retrain: a generic webhook, anything can call it.
  - Optional input-bucket polling (trigger.poll_input_bucket = true):
    periodically checks whether the configured input source's events file
    has changed (new S3 LastModified / local mtime) and triggers a run if
    so. Real S3 "event notifications" only deliver to SQS/SNS/Lambda, and
    non-AWS S3-compatible backends (R2, MinIO, ...) support them
    differently or not at all -- polling is the portable option that works
    the same way for every backend cicerone already supports, with no new
    required infra.

Both paths always trigger the exact same thing: one full, ordinary
job.run() call, identical to a scheduled run -- no continuous/online
retraining. RunGuard debounces so at most one run happens at a time and
rapid-fire triggers within a short window are coalesced into a no-op.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from fastapi import Depends, FastAPI, HTTPException

from cicerone import job
from cicerone.config import IOSettings, Settings
from cicerone.http_auth import require_bearer_token
from cicerone.io.options import build_s3_client, require_option

logger = logging.getLogger(__name__)


class _RunFn(Protocol):
    def __call__(self, triggered_by: str) -> None: ...


class RunGuard:
    """Debounces/guards concurrent or rapid-fire triggers: at most one
    training run happens at a time, and a trigger arriving less than
    `debounce_seconds` after the previous one started is ignored."""

    def __init__(self, debounce_seconds: float, run_fn: _RunFn = job.run):
        self._debounce_seconds = debounce_seconds
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self._running = False
        self._last_started_at: float | None = None

    def trigger(self, triggered_by: str) -> bool:
        """Attempts to start a run. Returns True if a run was started, False
        if the trigger was ignored (already running, or within the debounce
        window of the previous run)."""
        with self._lock:
            now = time.monotonic()
            if self._running:
                logger.info("Ignoring %s trigger: a run is already in progress", triggered_by)
                return False
            if self._last_started_at is not None and now - self._last_started_at < self._debounce_seconds:
                logger.info("Ignoring %s trigger: within debounce window", triggered_by)
                return False
            self._running = True
            self._last_started_at = now

        threading.Thread(target=self._run, args=(triggered_by,), daemon=True).start()
        return True

    def _run(self, triggered_by: str) -> None:
        try:
            self._run_fn(triggered_by=triggered_by)
        except Exception:
            logger.exception("Triggered run (%s) failed", triggered_by)
        finally:
            with self._lock:
                self._running = False


def _current_marker(input_settings: IOSettings) -> str | None:
    """A cheap "has the input data changed" fingerprint: local file mtime,
    or S3 object LastModified. Returns None if it can't be determined (e.g.
    file/object doesn't exist yet), which never counts as a change."""
    options = input_settings.options
    backend = options.get("storage_backend", "local")
    if backend == "local":
        path = Path(require_option(options, "path", "local")) / "events.parquet"
        return str(path.stat().st_mtime) if path.exists() else None

    try:
        client = build_s3_client(options)
        bucket = require_option(options, "bucket", "s3")
        prefix = str(options.get("prefix", "")).strip("/")
        key = f"{prefix}/events.parquet" if prefix else "events.parquet"
        head = client.head_object(Bucket=bucket, Key=key)
        return str(head["LastModified"])
    except Exception:
        logger.exception("Failed to check input source for changes")
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
        if marker is not None and marker != last_marker:
            last_marker = marker
            guard.trigger("s3-poll")


def create_app(settings: Settings, guard: RunGuard) -> FastAPI:
    app = FastAPI(title="cicerone-trigger")
    auth = require_bearer_token(settings.trigger_auth_token) if settings.trigger_auth_token else None
    dependencies = [Depends(auth)] if auth else []

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
