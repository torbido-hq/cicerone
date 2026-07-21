#!/usr/bin/env sh
set -eu

# Runs the recommendation job immediately, then keeps the process alive and
# re-triggers it according to CRON_SCHEDULE (default: daily at 03:00).
# This is a single low-volume batch job, so a tiny in-process scheduler is
# enough — no system cron daemon, no extra process manager.

echo "[entrypoint] running initial job..."
python -m cicerone.job

echo "[entrypoint] entering schedule loop (CRON_SCHEDULE=${CRON_SCHEDULE:-0 3 * * *})"
exec python -m cicerone.scheduler
