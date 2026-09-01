"""Public config API (``from cicerone.config import …``)."""

from __future__ import annotations

import importlib
from typing import Any

from cicerone.config.constants import (
    AUTOML_DEFAULT_N_SPLITS,
    AUTOML_DEFAULT_PRIMARY_METRIC,
    AUTOML_DEFAULT_TEST_DAYS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    DEFAULT_EPOCH_METRICS_EVERY,
    DEFAULT_EPOCH_METRICS_MAX_USERS,
    DEFAULT_EPOCH_METRICS_PLATEAU_EPS,
    DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW,
    DEFAULT_EPOCH_METRICS_REGRESSION_DROP,
    DEFAULT_EVENTS_BATCH_SIZE,
    DEFAULT_EVENTS_BATCH_WINDOW_SECONDS,
    DEFAULT_EVENTS_HEARTBEAT_SECONDS,
    DEFAULT_EVENTS_ONLINE_FIT_MIN_EVENTS,
    DEFAULT_EVENTS_ONLINE_FIT_PARTIAL_EPOCHS,
    DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS,
    DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
    DEFAULT_EVENTS_WEBHOOK_MAX_PENDING,
    DEFAULT_EXPERIMENT_ALPHA,
    DEFAULT_EXPLAIN_MAX_ATTRIBUTES,
    DEFAULT_EXPLAIN_MAX_SIMILAR_ITEMS,
    DEFAULT_ITEM_BASED_K_NEIGHBORS,
    DEFAULT_LOCK_KEY,
    DEFAULT_LOCK_TTL_SECONDS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODELS,
    DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS,
    EXPERIMENT_COMBINERS,
    LOCK_BACKENDS,
    MODES,
    PRIMARY_METRIC_WEIGHTED,
    RRF_K,
    STRATEGY_NAMES,
    ConfigError,
    Mode,
    StrategyName,
)
from cicerone.config.lock_url import (
    POSTGRES_LOCK_URL_REQUIRED,
    require_postgres_lock_url_parts,
    resolve_postgres_lock_url,
    resolve_postgres_lock_url_parts,
)
from cicerone.config.settings import (
    AutomlSettings,
    DashboardSettings,
    EpochMetricsSettings,
    EventsIncrementalSettings,
    EventsOnlineSettings,
    EventsSettings,
    ExperimentSettings,
    ExplainSettings,
    IOSettings,
    ServeSettings,
    Settings,
    TriggerSettings,
    VariantSettings,
)
from cicerone.config.validation import (
    resolve_epoch_metrics,
    resolve_max_workers,
    validate_model_weights,
    validate_rrf_k,
)

__all__ = [
    "AUTOML_DEFAULT_N_SPLITS",
    "AUTOML_DEFAULT_PRIMARY_METRIC",
    "AUTOML_DEFAULT_TEST_DAYS",
    "AutomlSettings",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS",
    "DEFAULT_EPOCH_METRICS_EVERY",
    "DEFAULT_EPOCH_METRICS_MAX_USERS",
    "DEFAULT_EPOCH_METRICS_PLATEAU_EPS",
    "DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW",
    "DEFAULT_EPOCH_METRICS_REGRESSION_DROP",
    "DEFAULT_EVENTS_BATCH_SIZE",
    "DEFAULT_EVENTS_BATCH_WINDOW_SECONDS",
    "DEFAULT_EVENTS_HEARTBEAT_SECONDS",
    "DEFAULT_EVENTS_ONLINE_FIT_MIN_EVENTS",
    "DEFAULT_EVENTS_ONLINE_FIT_PARTIAL_EPOCHS",
    "DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS",
    "DEFAULT_EVENTS_POLL_INTERVAL_SECONDS",
    "DEFAULT_EVENTS_WEBHOOK_MAX_PENDING",
    "DEFAULT_EXPERIMENT_ALPHA",
    "DEFAULT_EXPLAIN_MAX_ATTRIBUTES",
    "DEFAULT_EXPLAIN_MAX_SIMILAR_ITEMS",
    "DEFAULT_ITEM_BASED_K_NEIGHBORS",
    "DEFAULT_LOCK_KEY",
    "DEFAULT_LOCK_TTL_SECONDS",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MODELS",
    "DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS",
    "DashboardSettings",
    "EpochMetricsSettings",
    "EXPERIMENT_COMBINERS",
    "EventsIncrementalSettings",
    "EventsOnlineSettings",
    "EventsSettings",
    "ExperimentSettings",
    "ExplainSettings",
    "IOSettings",
    "LOCK_BACKENDS",
    "MODES",
    "Mode",
    "PRIMARY_METRIC_WEIGHTED",
    "POSTGRES_LOCK_URL_REQUIRED",
    "RRF_K",
    "STRATEGY_NAMES",
    "ServeSettings",
    "Settings",
    "StrategyName",
    "TriggerSettings",
    "VariantSettings",
    "coerce_events_settings",
    "load_events_settings",
    "load_experiment_settings",
    "load_settings",
    "make_settings",
    "require_postgres_lock_url_parts",
    "resolve_epoch_metrics",
    "resolve_max_workers",
    "resolve_postgres_lock_url",
    "resolve_postgres_lock_url_parts",
    "validate_model_weights",
    "validate_rrf_k",
]

_LAZY: dict[str, tuple[str, str]] = {
    "coerce_events_settings": ("cicerone.config.events", "coerce_events_settings"),
    "load_events_settings": ("cicerone.config.events", "load_events_settings"),
    "load_experiment_settings": ("cicerone.config.load", "load_experiment_settings"),
    "load_settings": ("cicerone.config.load", "load_settings"),
    "make_settings": ("cicerone.config.load", "make_settings"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value
