"""Config constants and mode aliases (no I/O)."""

from __future__ import annotations

from typing import Literal

DEFAULT_CONFIG_PATH = "/app/config/cicerone.toml"

# Kept here (not cicerone.model) so Settings.models validates without ML deps.
STRATEGY_NAMES: tuple[str, ...] = (
    "collaborative",
    "item_based",
    "sequential",
    "content_fallback",
    "popular",
    "latest",
)
DEFAULT_MODELS: list[str] = ["collaborative", "item_based", "popular"]
# Reciprocal rank fusion constant (Cormack et al., 2009); default for rrf_k.
RRF_K = 60

AUTOML_DEFAULT_N_SPLITS = 2
AUTOML_DEFAULT_TEST_DAYS = 14
AUTOML_DEFAULT_PRIMARY_METRIC = "MAP"
# Default 1: ProcessPool after threaded I/O deadlocks (LightFM/OpenBLAS).
DEFAULT_MAX_WORKERS = 1
DEFAULT_EPOCH_METRICS_EVERY = 5
DEFAULT_EPOCH_METRICS_MAX_USERS = 500
DEFAULT_EPOCH_METRICS_REGRESSION_DROP = 0.25
DEFAULT_EPOCH_METRICS_PLATEAU_EPS = 0.01
DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW = 3
# Canonical item-KNN neighbor default; also used by model_config RecTools K.
DEFAULT_ITEM_BASED_K_NEIGHBORS = 20
DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS = 50
# AutoML skip: median distinct items/user below this excludes sequential.
DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS = 5
DEFAULT_EXPLAIN_MAX_SIMILAR_ITEMS = 3
DEFAULT_EXPLAIN_MAX_ATTRIBUTES = 5

Mode = Literal["batch", "serve"]
# Strategy names stay as ``str`` and are validated against STRATEGY_NAMES at load.
StrategyName = str

MODES: tuple[Mode, ...] = ("batch", "serve")
LOCK_BACKENDS: tuple[str, ...] = ("in_process", "postgres", "redis")
DEFAULT_LOCK_KEY = "cicerone:scheduler:run_guard"
DEFAULT_LOCK_TTL_SECONDS = 24 * 60 * 60
# Apply flushes are short; a 24h retrain TTL would block events for a day after a crash.
DEFAULT_EVENTS_APPLY_LOCK_TTL_SECONDS = 60.0
# Cache the early retrain probe; the pre-write check is always live.
DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS = 1.0
DEFAULT_EVENTS_BATCH_SIZE = 100
DEFAULT_EVENTS_BATCH_WINDOW_SECONDS = 60.0
DEFAULT_EVENTS_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_EVENTS_WEBHOOK_MAX_PENDING = 10_000
DEFAULT_EVENTS_MAX_BODY_BYTES = 1_048_576
DEFAULT_SERVE_MAX_K = 100
DEFAULT_EVENTS_ONLINE_FIT_PARTIAL_EPOCHS = 1
DEFAULT_EVENTS_ONLINE_FIT_MIN_EVENTS = 100
# Cap online-only interaction rows appended on top of the last job artifact.
DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS = 50_000
# SQS apply visibility is 300s; beat well inside that window.
DEFAULT_EVENTS_HEARTBEAT_SECONDS = 15.0
EXPERIMENT_COMBINERS: tuple[str, ...] = ("priority", "rrf", "blend")
DEFAULT_EXPERIMENT_ALPHA = 0.05
PRIMARY_METRIC_WEIGHTED = "weighted"
PRIMARY_METRIC_CTR = "ctr"
PRIMARY_METRIC_CONVERSION = "conversion"
ATTRIBUTION_USER = "user"
ATTRIBUTION_CLICK = "click"
ATTRIBUTION_IMPRESSION = "impression"
ATTRIBUTION_RECOMMENDED = "recommended"
EXPERIMENT_ATTRIBUTIONS: tuple[str, ...] = (
    ATTRIBUTION_USER,
    ATTRIBUTION_CLICK,
    ATTRIBUTION_IMPRESSION,
    ATTRIBUTION_RECOMMENDED,
)
TRACK_KIND_IMPRESSION = "impression"
TRACK_KIND_CLICK = "click"
TRACK_KINDS: tuple[str, ...] = (TRACK_KIND_IMPRESSION, TRACK_KIND_CLICK)
DEFAULT_TRACK_ATTRIBUTION_WINDOW_HOURS = 24.0
DEFAULT_TRACK_MIN_IMPRESSIONS = 100
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ConfigError(ValueError):
    """Invalid config content or knobs (not missing files / unset env vars)."""
