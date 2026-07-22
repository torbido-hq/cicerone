"""Tiny in-process cron replacement.

Low-volume batch job: no need for a system cron daemon or a task queue.
Computes the next run time from the [job].cron_schedule set in cicerone.toml
(5-field cron expression), sleeps until then, runs the job, and repeats
forever. A single failed run is logged and does not crash the loop — the
next scheduled run still fires.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from croniter import croniter

from cicerone import job
from cicerone.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _seconds_until_next_run(schedule: str, now: datetime) -> float:
    next_run = croniter(schedule, now).get_next(datetime)
    return max((next_run - now).total_seconds(), 0)


def main() -> None:
    schedule = load_settings().cron_schedule
    if not croniter.is_valid(schedule):
        raise RuntimeError(f"Invalid cron_schedule: {schedule!r}")

    while True:
        now = datetime.now(UTC)
        sleep_seconds = _seconds_until_next_run(schedule, now)
        logger.info("Next run scheduled in %.0fs", sleep_seconds)
        time.sleep(sleep_seconds)

        try:
            job.run()
        except Exception:
            logger.exception("Scheduled run failed; will retry at the next scheduled time")


if __name__ == "__main__":
    main()
