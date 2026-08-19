"""Prometheus metrics for serve mode (default global registry)."""

from __future__ import annotations

import logging
import time

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

REQUESTS_TOTAL = Counter(
    "cicerone_requests_total",
    "HTTP requests handled by the serve process",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY_SECONDS = Histogram(
    "cicerone_request_latency_seconds",
    "HTTP request latency for serve routes",
    ["endpoint"],
)
CACHE_HITS_TOTAL = Counter(
    "cicerone_cache_hits_total",
    "Recommendation lookups that returned rows for the requested user",
)
CACHE_MISSES_TOTAL = Counter(
    "cicerone_cache_misses_total",
    "Recommendation lookups with no rows for the requested user",
)
CACHE_AGE_SECONDS = Gauge(
    "cicerone_cache_age_seconds",
    "Seconds since the last successful recommendations cache refresh",
)
CACHE_REFRESH_TOTAL = Counter(
    "cicerone_cache_refresh_total",
    "Background cache refresh outcomes",
    ["status"],
)
CACHE_REFRESH_DURATION_SECONDS = Histogram(
    "cicerone_cache_refresh_duration_seconds",
    "Time taken by a background cache refresh",
)
RETRAIN_TRIGGER_TOTAL = Counter(
    "cicerone_retrain_trigger_total",
    "Retrain trigger attempts (per replica)",
    ["source", "status"],
)
RECOMMENDATIONS_SERVED_TOTAL = Counter(
    "cicerone_recommendations_served_total",
    "Recommendation source tiers used to satisfy a request",
    ["source"],
)
EVENTS_SOURCE_CONNECTED = Gauge(
    "cicerone_events_source_connected",
    "Event source connectivity (1 connected, 0 otherwise); 0 when events disabled",
)
EVENTS_SOURCE_LAG = Gauge(
    "cicerone_events_source_lag",
    "Event source backlog behind the consume cursor; -1 when unknown or events disabled",
)
EVENTS_FLUSH_TOTAL = Counter(
    "cicerone_events_flush_total",
    "Incremental micro-batch flush outcomes",
    ["status"],
)
EVENTS_FLUSH_EVENTS_TOTAL = Counter(
    "cicerone_events_flush_events_total",
    "Events successfully applied by incremental flushes",
)
EVENTS_LAST_SUCCESS_TIMESTAMP_SECONDS = Gauge(
    "cicerone_events_last_success_timestamp_seconds",
    "Unix time of the last successful incremental flush (0 if never)",
)
EVENTS_TICK_ERRORS_TOTAL = Counter(
    "cicerone_events_tick_errors_total",
    "Unhandled exceptions in the event worker tick loop",
)
EVENTS_LOCK_TOTAL = Counter(
    "cicerone_events_lock_total",
    "Incremental apply-lease acquire attempts",
    ["status"],
)
EVENTS_LEADER = Gauge(
    "cicerone_events_leader",
    "1 when this replica currently owns the incremental apply lock in HA mode (0 otherwise)",
)
EVENTS_APPLY_BUSY_TOTAL = Counter(
    "cicerone_events_apply_busy_total",
    "Incremental flushes skipped because a lock was busy",
    ["reason"],
)
UP = Gauge("cicerone_up", "Serve process liveness (always 1 while running)")
UP.set(1)

METRICS_TOKEN_HEADER = "X-Metrics-Token"

_RETRAIN_SOURCES = frozenset({"webhook", "poll", "cron", "s3-poll", "manual"})
_EVENTS_FLUSH_STATUSES = frozenset({"success", "busy", "error"})
_EVENTS_LOCK_STATUSES = frozenset({"acquired", "skip"})
_EVENTS_APPLY_BUSY_REASONS = frozenset({"lock", "retrain"})

_SOURCE_TO_METRIC: dict[str, str] = {
    "personalized": "collaborative",
    "blended": "collaborative",
    "item_based": "item_based",
    "sequential": "sequential",
    "popular_fallback": "popular",
    "latest": "latest",
}

_last_successful_refresh_at: float | None = None

# Default gauges when events are off / not yet refreshed by the worker.
EVENTS_SOURCE_CONNECTED.set(0)
EVENTS_SOURCE_LAG.set(-1)
EVENTS_LAST_SUCCESS_TIMESTAMP_SECONDS.set(0)
EVENTS_LEADER.set(0)


def record_cache_hit() -> None:
    CACHE_HITS_TOTAL.inc()


def record_cache_miss() -> None:
    CACHE_MISSES_TOTAL.inc()


def observe_cache_refresh(*, duration_seconds: float, success: bool) -> None:
    global _last_successful_refresh_at
    CACHE_REFRESH_TOTAL.labels(status="success" if success else "failure").inc()
    CACHE_REFRESH_DURATION_SECONDS.observe(duration_seconds)
    if success:
        _last_successful_refresh_at = time.time()


def update_cache_age_gauge() -> None:
    if _last_successful_refresh_at is None:
        CACHE_AGE_SECONDS.set(0)
        return
    CACHE_AGE_SECONDS.set(max(0.0, time.time() - _last_successful_refresh_at))


def record_retrain_trigger(triggered_by: str, *, accepted: bool) -> None:
    source = triggered_by if triggered_by in _RETRAIN_SOURCES else "other"
    status = "accepted" if accepted else "debounced"
    RETRAIN_TRIGGER_TOTAL.labels(source=source, status=status).inc()


def record_recommendations_served(stored_sources: set[str]) -> None:
    seen: set[str] = set()
    for stored in stored_sources:
        metric_source = _SOURCE_TO_METRIC.get(stored)
        if metric_source is None or metric_source in seen:
            continue
        seen.add(metric_source)
        RECOMMENDATIONS_SERVED_TOTAL.labels(source=metric_source).inc()


def record_events_flush(*, status: str, events: int = 0) -> None:
    if status in _EVENTS_FLUSH_STATUSES:
        label = status
    else:
        logger.warning("Unknown events flush status %r; recording as error", status)
        label = "error"
    EVENTS_FLUSH_TOTAL.labels(status=label).inc()
    if label == "success" and events > 0:
        EVENTS_FLUSH_EVENTS_TOTAL.inc(events)
        EVENTS_LAST_SUCCESS_TIMESTAMP_SECONDS.set(time.time())


def record_events_tick_error() -> None:
    EVENTS_TICK_ERRORS_TOTAL.inc()


def record_events_lock(*, status: str) -> None:
    if status not in _EVENTS_LOCK_STATUSES:
        logger.warning("Unknown events lock status %r; recording as skip", status)
        status = "skip"
    EVENTS_LOCK_TOTAL.labels(status=status).inc()


def update_events_leader(is_leader: bool) -> None:
    EVENTS_LEADER.set(1 if is_leader else 0)


def record_events_apply_busy(*, reason: str) -> None:
    if reason not in _EVENTS_APPLY_BUSY_REASONS:
        logger.warning("Unknown events apply busy reason %r; recording as lock", reason)
        reason = "lock"
    EVENTS_APPLY_BUSY_TOTAL.labels(reason=reason).inc()


def update_events_source_health(*, connected: bool, lag: int | None) -> None:
    EVENTS_SOURCE_CONNECTED.set(1 if connected else 0)
    EVENTS_SOURCE_LAG.set(-1 if lag is None else max(0, int(lag)))
