"""Prometheus metrics for serve mode (default global registry)."""

from __future__ import annotations

import time

from prometheus_client import Counter, Gauge, Histogram

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
UP = Gauge("cicerone_up", "Serve process liveness (always 1 while running)")
UP.set(1)

METRICS_TOKEN_HEADER = "X-Metrics-Token"

_SOURCE_TO_METRIC: dict[str, str] = {
    "personalized": "collaborative",
    "blended": "collaborative",
    "item_based": "item_based",
    "popular_fallback": "popular",
    "latest": "latest",
}

_last_successful_refresh_at: float | None = None


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
    source = "webhook" if triggered_by == "webhook" else "poll"
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
