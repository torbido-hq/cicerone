"""Config constants and mode aliases (no I/O)."""

from __future__ import annotations

from typing import Literal

DEFAULT_CONFIG_PATH = "/app/config/cicerone.toml"

# Kept here (not cicerone.model) so Settings.models validates without ML deps.
STRATEGY_NAMES: tuple[str, ...] = (
    "collaborative",
    "item_based",
    "content_fallback",
    "popular",
    "latest",
)

AUTOML_DEFAULT_N_SPLITS = 2
AUTOML_DEFAULT_TEST_DAYS = 14
AUTOML_DEFAULT_PRIMARY_METRIC = "MAP"
# Sequential default: ProcessPool after threaded I/O deadlocks easily (LightFM/OpenBLAS).
DEFAULT_MAX_WORKERS = 1
DEFAULT_EPOCH_METRICS_EVERY = 5
DEFAULT_EPOCH_METRICS_MAX_USERS = 500
DEFAULT_EPOCH_METRICS_REGRESSION_DROP = 0.25
DEFAULT_EPOCH_METRICS_PLATEAU_EPS = 0.01
DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW = 3
# Canonical item-KNN neighbor default; also used by model_config RecTools K.
DEFAULT_ITEM_BASED_K_NEIGHBORS = 20
DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS = 50

Mode = Literal["batch", "serve"]
# Strategy names stay as ``str`` and are validated against STRATEGY_NAMES at load.
StrategyName = str

MODES: tuple[Mode, ...] = ("batch", "serve")
LOCK_BACKENDS: tuple[str, ...] = ("in_process", "postgres", "redis")


class ConfigError(ValueError):
    """Invalid config content or knobs (not missing files / unset env vars)."""
