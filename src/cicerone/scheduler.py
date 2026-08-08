"""In-process cron loop; optionally embeds the retrain trigger in-process."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

import uvicorn
from croniter import croniter

from cicerone import job
from cicerone.config import Settings, load_settings
from cicerone.locks import LockBackend, build_lock_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _seconds_until_next_run(schedule: str, now: datetime) -> float:
    next_run = croniter(schedule, now).get_next(datetime)
    return max((next_run - now).total_seconds(), 0)


def _cron_loop(schedule: str, run: Callable[[], None]) -> None:
    while True:
        now = datetime.now(UTC)
        sleep_seconds = _seconds_until_next_run(schedule, now)
        logger.info("Next run scheduled in %.0fs", sleep_seconds)
        time.sleep(sleep_seconds)

        try:
            run()
        except Exception:
            logger.exception("Scheduled run failed; will retry at the next scheduled time")


def _cron_run_with_lock(backend: LockBackend) -> None:
    """Synchronous cron path: skip this tick if another replica holds the lock."""
    if not backend.acquire():
        logger.info("Skipping cron run: distributed lock held by another instance")
        return
    try:
        job.run(triggered_by="cron")
    finally:
        backend.release()


def main() -> None:
    settings = load_settings()
    schedule = settings.cron_schedule
    if not croniter.is_valid(schedule):
        raise RuntimeError(f"Invalid cron_schedule: {schedule!r}")

    if settings.trigger.enabled:
        _run_with_trigger(settings, schedule)
    elif settings.trigger.lock_backend != "in_process":
        backend = build_lock_backend(settings)
        _cron_loop(schedule, lambda: _cron_run_with_lock(backend))
    else:
        _cron_loop(schedule, lambda: job.run(triggered_by="cron"))


def _run_with_trigger(settings: Settings, schedule: str) -> None:
    from cicerone.trigger import RunGuard, create_app, poll_input_forever

    # Default in_process: omit backend so RunGuard uses its threading.Lock path.
    lock_backend = None if settings.trigger.lock_backend == "in_process" else build_lock_backend(settings)
    guard = RunGuard(
        settings.trigger.debounce_seconds,
        lock_backend=lock_backend,
    )
    threading.Thread(target=_cron_loop, args=(schedule, lambda: guard.trigger("cron")), daemon=True).start()

    if settings.trigger.poll_input_bucket:
        threading.Thread(
            target=poll_input_forever,
            args=(settings.input, guard, settings.trigger.poll_interval_seconds),
            daemon=True,
        ).start()

    app = create_app(settings, guard)
    uvicorn.run(app, host=settings.trigger.host, port=settings.trigger.port)


if __name__ == "__main__":
    main()
