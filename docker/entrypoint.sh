#!/usr/bin/env sh
set -eu

# Batch mode (default): runs the recommendation job immediately, then keeps
# the process alive and re-triggers it according to CRON_SCHEDULE (see
# cicerone.toml's [job].cron_schedule), plus an optional event-driven
# trigger (see [job.trigger]) -- all handled by cicerone.scheduler.
#
# Serve mode (job.mode = "serve" in cicerone.toml): runs the lightweight
# read API instead (cicerone.serve) -- no training job, no scheduler.
#
# This is a single low-volume batch job, so a tiny in-process scheduler is
# enough -- no system cron daemon, no extra process manager.

CICERONE_MODE="$(python -c 'from cicerone.config import load_settings; print(load_settings().mode)')"

if [ "$CICERONE_MODE" = "serve" ]; then
    echo "[entrypoint] mode=serve, starting read API..."
    exec python -m cicerone.serve
fi

echo "[entrypoint] mode=batch, running initial job..."
python -m cicerone.job

echo "[entrypoint] entering schedule loop (CRON_SCHEDULE=${CRON_SCHEDULE:-0 3 * * *})"
exec python -m cicerone.scheduler
