"""Configuration for the Cicerone recommender job.

Loaded from a TOML file (default /app/config/cicerone.toml, override with
CICERONE_CONFIG_PATH). Secrets use ${ENV_VAR_NAME} placeholders; escape a
literal "${...}" as "$${...}".

This package replaces the former monolithic ``config.py``; public imports stay
``from cicerone.config import …``.
"""

from __future__ import annotations

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
    DEFAULT_ITEM_BASED_K_NEIGHBORS,
    DEFAULT_MAX_WORKERS,
    MODES,
    STRATEGY_NAMES,
    ConfigError,
    Mode,
    StrategyName,
)
from cicerone.config.load import load_settings, make_settings
from cicerone.config.settings import (
    AutomlSettings,
    DashboardSettings,
    EpochMetricsSettings,
    IOSettings,
    ServeSettings,
    Settings,
    TriggerSettings,
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
    "DEFAULT_ITEM_BASED_K_NEIGHBORS",
    "DEFAULT_MAX_WORKERS",
    "DashboardSettings",
    "EpochMetricsSettings",
    "IOSettings",
    "MODES",
    "Mode",
    "STRATEGY_NAMES",
    "ServeSettings",
    "Settings",
    "StrategyName",
    "TriggerSettings",
    "load_settings",
    "make_settings",
    "resolve_epoch_metrics",
    "resolve_max_workers",
    "validate_model_weights",
    "validate_rrf_k",
]
