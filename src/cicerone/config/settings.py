"""Settings dataclasses for job / serve / trigger / dashboard / AutoML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cicerone.config.constants import (
    ALLOCATION_FIXED,
    AUTOML_DEFAULT_N_SPLITS,
    AUTOML_DEFAULT_PRIMARY_METRIC,
    AUTOML_DEFAULT_TEST_DAYS,
    DEFAULT_EPOCH_METRICS_MAX_USERS,
    DEFAULT_EPOCH_METRICS_PLATEAU_EPS,
    DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW,
    DEFAULT_EPOCH_METRICS_REGRESSION_DROP,
    DEFAULT_EVENTS_BATCH_SIZE,
    DEFAULT_EVENTS_BATCH_WINDOW_SECONDS,
    DEFAULT_EVENTS_ONLINE_FIT_MIN_EVENTS,
    DEFAULT_EVENTS_ONLINE_FIT_PARTIAL_EPOCHS,
    DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS,
    DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
    DEFAULT_EXPERIMENT_ALPHA,
    DEFAULT_EXPLAIN_MAX_ATTRIBUTES,
    DEFAULT_EXPLAIN_MAX_SIMILAR_ITEMS,
    DEFAULT_LOCK_KEY,
    DEFAULT_LOCK_TTL_SECONDS,
    DEFAULT_SERVE_CONSUMED_LOOKBACK,
    DEFAULT_THOMPSON_EXPLORE_TRAFFIC,
    DEFAULT_THOMPSON_ROTATE_MIN_PROB,
    DEFAULT_TRACK_ATTRIBUTION_WINDOW_HOURS,
    DEFAULT_TRACK_MIN_IMPRESSIONS,
    PRIMARY_METRIC_WEIGHTED,
    Mode,
)

if TYPE_CHECKING:
    from cicerone.feature_config import BoostRule, EligibilityRule


@dataclass(frozen=True)
class EpochMetricsSettings:
    """Tunables for optional collaborative/sequential per-epoch metric logging."""

    every: int
    max_users: int = DEFAULT_EPOCH_METRICS_MAX_USERS
    regression_drop: float = DEFAULT_EPOCH_METRICS_REGRESSION_DROP
    plateau_eps: float = DEFAULT_EPOCH_METRICS_PLATEAU_EPS
    plateau_window: int = DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW


@dataclass(frozen=True)
class ServeSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str | None = None
    default_k: int = 10
    refresh_interval_seconds: float = 60.0
    category_column: str = "category"
    metrics_enabled: bool = False
    metrics_token: str | None = None
    log_impressions: bool = False
    exclude_consumed: bool = True
    fallback_fill: bool = True
    consumed_lookback: int = DEFAULT_SERVE_CONSUMED_LOOKBACK


@dataclass(frozen=True)
class TriggerSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    auth_token: str | None = None
    debounce_seconds: float = 60.0
    poll_input_bucket: bool = False
    poll_input_single_replica: bool = False
    poll_interval_seconds: float = 300.0
    lock_backend: str = "in_process"
    postgres_url: str | None = None
    redis_url: str | None = None
    lock_key: str = DEFAULT_LOCK_KEY
    lock_ttl_seconds: float = float(DEFAULT_LOCK_TTL_SECONDS)


@dataclass(frozen=True)
class DashboardSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8090
    users_path: str = "/app/config/dashboard_users.toml"
    refresh_interval_seconds: float = 30.0
    history_limit: int = 20
    lookup_events: int = 20
    lookup_user_attrs: tuple[str, ...] = ()
    # Retrain webhook as the dashboard reaches it; unset derives a loopback URL from [job.trigger].
    trigger_url: str | None = None
    trigger_auth_token: str | None = None
    # Plain http to a non-loopback trigger_url sends the token in clear; only for a private network.
    trigger_allow_http: bool = False


@dataclass(frozen=True)
class AutomlSettings:
    enabled: bool = False
    n_splits: int = AUTOML_DEFAULT_N_SPLITS
    test_days: int = AUTOML_DEFAULT_TEST_DAYS
    primary_metric: str = AUTOML_DEFAULT_PRIMARY_METRIC
    candidates: list[dict[str, Any]] | None = None
    debias: bool = False


@dataclass(frozen=True)
class EventsIncrementalSettings:
    batch_size: int = DEFAULT_EVENTS_BATCH_SIZE
    batch_window_seconds: float = DEFAULT_EVENTS_BATCH_WINDOW_SECONDS
    poll_interval_seconds: float = DEFAULT_EVENTS_POLL_INTERVAL_SECONDS


@dataclass(frozen=True)
class EventsOnlineSettings:
    """Serve-worker LightFM fit_partial + user-scoped recommend write-through."""

    enabled: bool = False
    fit_partial_epochs: int = DEFAULT_EVENTS_ONLINE_FIT_PARTIAL_EPOCHS
    fit_min_events: int = DEFAULT_EVENTS_ONLINE_FIT_MIN_EVENTS
    max_extra_interactions: int = DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS


@dataclass(frozen=True)
class EventsSettings:
    enabled: bool = False
    kind: str = "webhook"
    options: dict[str, Any] = field(default_factory=dict)
    incremental: EventsIncrementalSettings = field(default_factory=EventsIncrementalSettings)
    ha: bool = False
    online: EventsOnlineSettings = field(default_factory=EventsOnlineSettings)


@dataclass(frozen=True)
class PublishSettings:
    enabled: bool = False
    kind: str = "kafka"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantSettings:
    """One ranking recipe in an ``[experiment]`` block."""

    name: str
    traffic: float
    models: list[str] | None = None
    model_weights: dict[str, float] | None = None
    rrf_k: float | None = None
    combiner: str | None = None
    blending: dict[str, Any] | None = None
    boosts: bool | tuple[str, ...] | tuple[BoostRule, ...] = True
    eligibility: bool | tuple[str, ...] | tuple[EligibilityRule, ...] = True


@dataclass(frozen=True)
class ExperimentSettings:
    enabled: bool = False
    id: str = ""
    primary_metric: str = PRIMARY_METRIC_WEIGHTED
    variants: tuple[VariantSettings, ...] = ()
    log_exposures: bool = False
    automl_challenger: bool = False
    alpha: float = DEFAULT_EXPERIMENT_ALPHA
    attribution: str = "user"
    allocation: str = ALLOCATION_FIXED
    explore_traffic: float = DEFAULT_THOMPSON_EXPLORE_TRAFFIC
    rotate_min_prob: float = DEFAULT_THOMPSON_ROTATE_MIN_PROB


@dataclass(frozen=True)
class TrackSettings:
    """Impression/click ingest; kept off the training event path."""

    enabled: bool = False
    attribution_window_hours: float = DEFAULT_TRACK_ATTRIBUTION_WINDOW_HOURS
    conversion_event_types: tuple[str, ...] = ()
    min_impressions: int = DEFAULT_TRACK_MIN_IMPRESSIONS
    redis_url: str | None = None
    # None: derived from the [output] location (see track.store_redis.default_redis_key_prefix).
    redis_key_prefix: str | None = None


@dataclass(frozen=True)
class EvalSettings:
    """Job-time production replay of previously written lists."""

    enabled: bool = False
    event_types: tuple[str, ...] = ()
    ks: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExplainSettings:
    """Batch-time recommendation reasons persisted on each output row."""

    enabled: bool = True
    max_similar_items: int = DEFAULT_EXPLAIN_MAX_SIMILAR_ITEMS
    max_attributes: int = DEFAULT_EXPLAIN_MAX_ATTRIBUTES


@dataclass(frozen=True)
class IOSettings:
    kind: str
    options: dict[str, Any] = field(default_factory=dict)
    artifact_hmac_key: str | None = None


@dataclass(frozen=True)
class Settings:
    input: IOSettings
    output: IOSettings
    feature_config_path: str
    top_k: int
    half_life_days: float
    cron_schedule: str
    models: list[str] | None
    model_weights: dict[str, float] | None
    rrf_k: float | None
    save_model_artifact: bool
    max_workers: int
    epoch_metrics: EpochMetricsSettings | None
    item_based_k_neighbors: int
    model_configs: dict[str, dict[str, Any]]  # RecTools configs; no content_fallback
    content_fallback_enabled: bool
    content_fallback_max_neighbors: int
    sequential_min_median_interactions: int
    automl: AutomlSettings
    mode: Mode
    serve: ServeSettings
    trigger: TriggerSettings
    dashboard: DashboardSettings
    events: EventsSettings
    publish: PublishSettings = field(default_factory=PublishSettings)
    explain: ExplainSettings = field(default_factory=ExplainSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    track: TrackSettings = field(default_factory=TrackSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
