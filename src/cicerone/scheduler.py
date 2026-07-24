"""Tiny in-process cron replacement.

Low-volume batch job: no need for a system cron daemon or a task queue.
Computes the next run time from the [job].cron_schedule set in cicerone.toml
(5-field cron expression), sleeps until then, runs the job, and repeats
forever. A single failed run is logged and does not crash the loop — the
next scheduled run still fires.

When [job.trigger].enabled is set, this also starts the event-driven
retrain trigger (cicerone.trigger: webhook + optional input-bucket poller)
in the same process, additive to the cron loop above -- both funnel through
the same RunGuard so at most one run happens at a time regardless of what
triggered it. Batch-only configs (trigger.enabled unset/false, the default)
behave exactly as before: no HTTP server, just the cron loop.
"""

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


def main() -> None:
    settings = load_settings()
    schedule = settings.cron_schedule
    if not croniter.is_valid(schedule):
        raise RuntimeError(f"Invalid cron_schedule: {schedule!r}")

    if settings.trigger_enabled:
        _run_with_trigger(settings, schedule)
    else:
        _cron_loop(schedule, lambda: job.run(triggered_by="cron"))


def _run_with_trigger(settings: Settings, schedule: str) -> None:
    from cicerone.trigger import RunGuard, create_app, poll_input_forever

    guard = RunGuard(settings.trigger_debounce_seconds)
    threading.Thread(target=_cron_loop, args=(schedule, lambda: guard.trigger("cron")), daemon=True).start()

    if settings.trigger_poll_input_bucket:
        threading.Thread(
            target=poll_input_forever,
            args=(settings.input, guard, settings.trigger_poll_interval_seconds),
            daemon=True,
        ).start()

    app = create_app(settings, guard)
    uvicorn.run(app, host=settings.trigger_host, port=settings.trigger_port)


if __name__ == "__main__":
    main()
