"""Settings dataclasses for job / serve / trigger / dashboard / AutoML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cicerone.config.constants import (
    AUTOML_DEFAULT_N_SPLITS,
    AUTOML_DEFAULT_PRIMARY_METRIC,
    AUTOML_DEFAULT_TEST_DAYS,
    DEFAULT_EPOCH_METRICS_MAX_USERS,
    DEFAULT_EPOCH_METRICS_PLATEAU_EPS,
    DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW,
    DEFAULT_EPOCH_METRICS_REGRESSION_DROP,
    DEFAULT_EVENTS_BATCH_SIZE,
    DEFAULT_EVENTS_BATCH_WINDOW_SECONDS,
    DEFAULT_LOCK_KEY,
    DEFAULT_LOCK_TTL_SECONDS,
    Mode,
)


@dataclass(frozen=True)
class EpochMetricsSettings:
    """Tunables for optional LightFM per-epoch metric logging."""

    every: int
    max_users: int = DEFAULT_EPOCH_METRICS_MAX_USERS
    regression_drop: float = DEFAULT_EPOCH_METRICS_REGRESSION_DROP
    plateau_eps: float = DEFAULT_EPOCH_METRICS_PLATEAU_EPS
    plateau_window: int = DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW


@dataclass(frozen=True)
class ServeSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    auth_token: str | None = None
    default_k: int = 10
    refresh_interval_seconds: float = 60.0
    category_column: str = "category"
    metrics_enabled: bool = True
    metrics_token: str | None = None


@dataclass(frozen=True)
class TriggerSettings:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    auth_token: str | None = None
    debounce_seconds: float = 60.0
    poll_input_bucket: bool = False
    poll_interval_seconds: float = 300.0
    lock_backend: str = "in_process"
    postgres_url: str | None = None
    redis_url: str | None = None
    lock_key: str = DEFAULT_LOCK_KEY
    lock_ttl_seconds: float = float(DEFAULT_LOCK_TTL_SECONDS)


@dataclass(frozen=True)
class DashboardSettings:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8090
    users_path: str = "/app/config/dashboard_users.toml"
    refresh_interval_seconds: float = 30.0
    history_limit: int = 20


@dataclass(frozen=True)
class AutomlSettings:
    enabled: bool = False
    n_splits: int = AUTOML_DEFAULT_N_SPLITS
    test_days: int = AUTOML_DEFAULT_TEST_DAYS
    primary_metric: str = AUTOML_DEFAULT_PRIMARY_METRIC
    candidates: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class EventsIncrementalSettings:
    batch_size: int = DEFAULT_EVENTS_BATCH_SIZE
    batch_window_seconds: float = DEFAULT_EVENTS_BATCH_WINDOW_SECONDS


@dataclass(frozen=True)
class EventsSettings:
    enabled: bool = False
    kind: str = "webhook"
    options: dict[str, Any] = field(default_factory=dict)
    incremental: EventsIncrementalSettings = field(default_factory=EventsIncrementalSettings)


@dataclass(frozen=True)
class IOSettings:
    kind: str
    options: dict[str, Any] = field(default_factory=dict)


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
    automl: AutomlSettings
    mode: Mode
    serve: ServeSettings
    trigger: TriggerSettings
    dashboard: DashboardSettings
    events: EventsSettings

    @property
    def serve_host(self) -> str:
        return self.serve.host

    @property
    def serve_port(self) -> int:
        return self.serve.port

    @property
    def serve_auth_token(self) -> str | None:
        return self.serve.auth_token

    @property
    def serve_default_k(self) -> int:
        return self.serve.default_k

    @property
    def serve_refresh_interval_seconds(self) -> float:
        return self.serve.refresh_interval_seconds

    @property
    def serve_category_column(self) -> str:
        return self.serve.category_column

    @property
    def serve_metrics_enabled(self) -> bool:
        return self.serve.metrics_enabled

    @property
    def serve_metrics_token(self) -> str | None:
        return self.serve.metrics_token

    @property
    def trigger_enabled(self) -> bool:
        return self.trigger.enabled

    @property
    def trigger_host(self) -> str:
        return self.trigger.host

    @property
    def trigger_port(self) -> int:
        return self.trigger.port

    @property
    def trigger_auth_token(self) -> str | None:
        return self.trigger.auth_token

    @property
    def trigger_debounce_seconds(self) -> float:
        return self.trigger.debounce_seconds

    @property
    def trigger_poll_input_bucket(self) -> bool:
        return self.trigger.poll_input_bucket

    @property
    def trigger_poll_interval_seconds(self) -> float:
        return self.trigger.poll_interval_seconds

    @property
    def trigger_lock_backend(self) -> str:
        return self.trigger.lock_backend

    @property
    def trigger_postgres_url(self) -> str | None:
        return self.trigger.postgres_url

    @property
    def trigger_redis_url(self) -> str | None:
        return self.trigger.redis_url

    @property
    def trigger_lock_key(self) -> str:
        return self.trigger.lock_key

    @property
    def trigger_lock_ttl_seconds(self) -> float:
        return self.trigger.lock_ttl_seconds

    @property
    def dashboard_enabled(self) -> bool:
        return self.dashboard.enabled

    @property
    def dashboard_host(self) -> str:
        return self.dashboard.host

    @property
    def dashboard_port(self) -> int:
        return self.dashboard.port

    @property
    def dashboard_users_path(self) -> str:
        return self.dashboard.users_path

    @property
    def dashboard_refresh_interval_seconds(self) -> float:
        return self.dashboard.refresh_interval_seconds

    @property
    def dashboard_history_limit(self) -> int:
        return self.dashboard.history_limit

    @property
    def automl_enabled(self) -> bool:
        return self.automl.enabled

    @property
    def automl_n_splits(self) -> int:
        return self.automl.n_splits

    @property
    def automl_test_days(self) -> int:
        return self.automl.test_days

    @property
    def automl_primary_metric(self) -> str:
        return self.automl.primary_metric

    @property
    def automl_candidates(self) -> list[dict[str, Any]] | None:
        return self.automl.candidates

    @property
    def events_enabled(self) -> bool:
        return self.events.enabled

    @property
    def events_kind(self) -> str:
        return self.events.kind
