from __future__ import annotations

from prometheus_client import REGISTRY, generate_latest
from prometheus_client.parser import text_string_to_metric_families

import cicerone.dashboard  # noqa: F401
import cicerone.job  # noqa: F401
import cicerone.scheduler  # noqa: F401
import cicerone.serve  # noqa: F401
import cicerone.trigger  # noqa: F401

EXPECTED_FAMILIES = {
    "cicerone_cache_age_seconds": ("gauge", []),
    "cicerone_cache_hits": ("counter", []),
    "cicerone_cache_misses": ("counter", []),
    "cicerone_cache_refresh": ("counter", ["status"]),
    "cicerone_cache_refresh_duration_seconds": ("histogram", []),
    "cicerone_events_apply_busy": ("counter", ["reason"]),
    "cicerone_events_flush": ("counter", ["status"]),
    "cicerone_events_flush_events": ("counter", []),
    "cicerone_events_last_success_timestamp_seconds": ("gauge", []),
    "cicerone_events_leader": ("gauge", []),
    "cicerone_events_lock": ("counter", ["status"]),
    "cicerone_events_replayed": ("counter", []),
    "cicerone_events_source_connected": ("gauge", []),
    "cicerone_events_source_lag": ("gauge", []),
    "cicerone_events_tick_errors": ("counter", []),
    "cicerone_recommendations_experiment": ("counter", ["experiment_id", "variant"]),
    "cicerone_recommendations_served": ("counter", ["source"]),
    "cicerone_request_latency_seconds": ("histogram", ["endpoint"]),
    "cicerone_requests": ("counter", ["endpoint", "method", "status"]),
    "cicerone_retrain_trigger": ("counter", ["source", "status"]),
    "cicerone_serve_appends_dropped": ("counter", ["kind"]),
    "cicerone_track_ingest": ("counter", ["kind", "status"]),
    "cicerone_up": ("gauge", []),
}


def test_exposed_metric_families_and_types():
    families = {
        family.name: family.type
        for family in text_string_to_metric_families(generate_latest().decode())
        # The text parser reports prometheus_client's ``*_created`` samples as separate gauges.
        if family.name.startswith("cicerone_") and not family.name.endswith("_created")
    }
    assert families == {name: kind for name, (kind, _labels) in EXPECTED_FAMILIES.items()}


def test_metric_label_names():
    labels: dict[str, list[str]] = {}
    # prometheus_client has no public API for a family's declared label names.
    for collector in list(REGISTRY._collector_to_names):
        for family in getattr(collector, "describe", collector.collect)():
            if family.name.startswith("cicerone_"):
                labels[family.name] = sorted(getattr(collector, "_labelnames", ()))
    assert labels == {name: names for name, (_kind, names) in EXPECTED_FAMILIES.items()}
