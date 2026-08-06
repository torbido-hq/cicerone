"""Configuration for the Cicerone recommender job.

Loaded from a TOML file (default /app/config/cicerone.toml, override with
CICERONE_CONFIG_PATH). Secrets use ${ENV_VAR_NAME} placeholders; escape a
literal "${...}" as "$${...}".
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

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
# Default sequential: ProcessPool after threaded I/O (and with LightFM/OpenBLAS)
# deadlocks easily in Docker/CI when forked from a multithreaded parent.
DEFAULT_MAX_WORKERS = 1
DEFAULT_EPOCH_METRICS_EVERY = 5
DEFAULT_EPOCH_METRICS_MAX_USERS = 500
DEFAULT_EPOCH_METRICS_REGRESSION_DROP = 0.25
DEFAULT_EPOCH_METRICS_PLATEAU_EPS = 0.01
DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW = 3
DEFAULT_ITEM_BASED_K_NEIGHBORS = 20
DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS = 50

Mode = Literal["batch", "serve"]
# Strategy names stay as ``str`` and are validated against STRATEGY_NAMES at load.
StrategyName = str

MODES: tuple[Mode, ...] = ("batch", "serve")


class ConfigError(ValueError):
    """Invalid config content or knobs (not missing files / unset env vars)."""


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


@dataclass(frozen=True)
class TriggerSettings:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    auth_token: str | None = None
    debounce_seconds: float = 60.0
    poll_input_bucket: bool = False
    poll_interval_seconds: float = 300.0


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


def resolve_max_workers(raw: Any | None = None) -> int:
    """Process-pool size for AutoML folds / strategy fitting.

    Omit or pass ``None`` for sequential (``1``). An explicit integer must be >= 1.
    """
    if raw is None:
        return DEFAULT_MAX_WORKERS
    workers = int(raw)
    if workers < 1:
        raise ConfigError(f"job.max_workers must be >= 1, got {workers}")
    return workers


def resolve_epoch_metrics(
    *,
    log_epoch_metrics: bool,
    every: Any | None = None,
    max_users: Any | None = None,
    regression_drop: Any | None = None,
    plateau_eps: Any | None = None,
    plateau_window: Any | None = None,
) -> EpochMetricsSettings | None:
    """Build epoch-metric settings, or ``None`` when logging is off.

    Interval / threshold knobs are validated only when logging is enabled.
    """
    if not log_epoch_metrics:
        return None
    return EpochMetricsSettings(
        every=_require_positive_int(
            DEFAULT_EPOCH_METRICS_EVERY if every is None else int(every),
            name="job.epoch_metrics_every",
        ),
        max_users=_require_positive_int(
            DEFAULT_EPOCH_METRICS_MAX_USERS if max_users is None else int(max_users),
            name="job.epoch_metrics_max_users",
        ),
        regression_drop=_require_unit_interval(
            DEFAULT_EPOCH_METRICS_REGRESSION_DROP if regression_drop is None else float(regression_drop),
            name="job.epoch_metrics_regression_drop",
        ),
        plateau_eps=_require_unit_interval(
            DEFAULT_EPOCH_METRICS_PLATEAU_EPS if plateau_eps is None else float(plateau_eps),
            name="job.epoch_metrics_plateau_eps",
        ),
        plateau_window=_require_positive_int(
            DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW if plateau_window is None else int(plateau_window),
            name="job.epoch_metrics_plateau_window",
        ),
    )


def validate_model_weights(weights: dict[str, float] | None, *, context: str = "model_weights") -> None:
    if weights is None:
        return
    negative_weights = {name: weight for name, weight in weights.items() if weight < 0}
    if negative_weights:
        raise ConfigError(f"{context} value(s) must be non-negative, got {negative_weights}")


def validate_rrf_k(rrf_k: float | None, *, context: str = "rrf_k") -> None:
    if rrf_k is not None and rrf_k <= 0:
        raise ConfigError(f"{context} must be positive, got {rrf_k}")


def _require_positive_int(value: int, *, name: str) -> int:
    if value < 1:
        raise ConfigError(f"{name} must be >= 1, got {value}")
    return value


def _require_positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value


def _require_unit_interval(value: float, *, name: str) -> float:
    """Require a relative fraction in (0, 1]."""
    if value <= 0 or value > 1:
        raise ConfigError(f"{name} must be in (0, 1], got {value}")
    return value


_ENV_PLACEHOLDER = re.compile(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_placeholders(value: Any, path: str = "") -> Any:
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            escaped, name = match.group(1), match.group(2)
            if escaped:
                return f"${{{name}}}"
            if name not in os.environ:
                location = f" (at '{path}')" if path else ""
                raise RuntimeError(
                    f"Config references ${{{name}}}{location} but that environment variable is not set"
                )
            return os.environ[name]

        return _ENV_PLACEHOLDER.sub(_replace, value)
    if isinstance(value, dict):
        return {
            key: _resolve_env_placeholders(item, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_env_placeholders(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


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
    content_fallback_enabled: bool
    content_fallback_max_neighbors: int
    automl: AutomlSettings
    mode: Mode
    serve: ServeSettings
    trigger: TriggerSettings
    dashboard: DashboardSettings

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


_SERVE_FLAT_KEYS = (
    ("serve_host", "host"),
    ("serve_port", "port"),
    ("serve_auth_token", "auth_token"),
    ("serve_default_k", "default_k"),
    ("serve_refresh_interval_seconds", "refresh_interval_seconds"),
    ("serve_category_column", "category_column"),
)
_TRIGGER_FLAT_KEYS = (
    ("trigger_enabled", "enabled"),
    ("trigger_host", "host"),
    ("trigger_port", "port"),
    ("trigger_auth_token", "auth_token"),
    ("trigger_debounce_seconds", "debounce_seconds"),
    ("trigger_poll_input_bucket", "poll_input_bucket"),
    ("trigger_poll_interval_seconds", "poll_interval_seconds"),
)
_DASHBOARD_FLAT_KEYS = (
    ("dashboard_enabled", "enabled"),
    ("dashboard_host", "host"),
    ("dashboard_port", "port"),
    ("dashboard_users_path", "users_path"),
    ("dashboard_refresh_interval_seconds", "refresh_interval_seconds"),
    ("dashboard_history_limit", "history_limit"),
)
_AUTOML_FLAT_KEYS = (
    ("automl_enabled", "enabled"),
    ("automl_n_splits", "n_splits"),
    ("automl_test_days", "test_days"),
    ("automl_primary_metric", "primary_metric"),
    ("automl_candidates", "candidates"),
)


def _coerce_nested(
    cls: type[Any],
    nested: Any | None,
    flat_keys: tuple[tuple[str, str], ...],
    overrides: dict[str, Any],
) -> Any:
    """Build a nested settings object from an optional nested value + flat kwargs."""
    if isinstance(nested, cls):
        base = nested
    elif isinstance(nested, dict):
        base = cls(**nested)
    elif nested is None:
        base = cls()
    else:
        raise TypeError(f"Expected {cls.__name__}, dict, or None; got {type(nested).__name__}")

    updates: dict[str, Any] = {}
    for flat_key, field_name in flat_keys:
        if flat_key in overrides:
            updates[field_name] = overrides.pop(flat_key)
    return replace(base, **updates) if updates else base


def make_settings(**overrides: Any) -> Settings:
    """Build a fully-populated ``Settings`` with the same defaults as a typical TOML load.

    Used by tests and OpenAPI schema export so callers only pass the fields they
    care about (``mode``, auth tokens, …) and stay aligned as ``Settings`` grows.
    Prefer ``load_settings(path)`` for real config files.

    Accepts flat kwargs (``serve_host=…``, ``automl_enabled=…``) and/or nested
    objects (``serve=ServeSettings(…)``).
    """
    serve = _coerce_nested(ServeSettings, overrides.pop("serve", None), _SERVE_FLAT_KEYS, overrides)
    trigger = _coerce_nested(TriggerSettings, overrides.pop("trigger", None), _TRIGGER_FLAT_KEYS, overrides)
    dashboard = _coerce_nested(
        DashboardSettings, overrides.pop("dashboard", None), _DASHBOARD_FLAT_KEYS, overrides
    )
    automl = _coerce_nested(AutomlSettings, overrides.pop("automl", None), _AUTOML_FLAT_KEYS, overrides)

    base: dict[str, Any] = dict(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/in"}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
        feature_config_path="/app/config/features.toml",
        top_k=10,
        half_life_days=90,
        cron_schedule="0 3 * * *",
        models=None,
        model_weights=None,
        rrf_k=None,
        save_model_artifact=False,
        max_workers=DEFAULT_MAX_WORKERS,
        epoch_metrics=None,
        item_based_k_neighbors=DEFAULT_ITEM_BASED_K_NEIGHBORS,
        content_fallback_enabled=False,
        content_fallback_max_neighbors=DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
        automl=automl,
        mode="batch",
        serve=serve,
        trigger=trigger,
        dashboard=dashboard,
    )
    base.update(overrides)
    return Settings(**base)


def _load_io_settings(raw: dict[str, Any], section_name: str) -> IOSettings:
    section = raw.get(section_name)
    if not section:
        raise ConfigError(f"Missing required config section: [{section_name}]")
    if "kind" not in section:
        raise ConfigError(f"Missing required config key: [{section_name}].kind")
    options = _resolve_env_placeholders(section.get("options", {}), f"{section_name}.options")
    return IOSettings(kind=str(section["kind"]).lower(), options=options)


def load_settings(config_path: str | None = None) -> Settings:
    path = Path(config_path or os.environ.get("CICERONE_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    job = raw.get("job", {})
    automl = job.get("automl", {})
    models = list(job["models"]) if "models" in job else None
    if models is not None:
        if not models:
            raise ConfigError(
                "job.models is empty; configure at least one model name, or omit job.models entirely "
                "to use the default"
            )
        unknown_models = [name for name in models if name not in STRATEGY_NAMES]
        if unknown_models:
            raise ConfigError(
                f"job.models contains unknown model(s) {unknown_models}; available: {list(STRATEGY_NAMES)}"
            )
    model_weights = (
        {name: float(weight) for name, weight in job["model_weights"].items()}
        if "model_weights" in job
        else None
    )
    validate_model_weights(model_weights, context="job.model_weights")
    if model_weights is not None and models is not None:
        unknown_weights = [name for name in model_weights if name not in models]
        if unknown_weights:
            raise ConfigError(f"job.model_weights key(s) {unknown_weights} are not in job.models {models}")
    rrf_k = float(job["rrf_k"]) if "rrf_k" in job else None
    validate_rrf_k(rrf_k, context="job.rrf_k")

    mode = str(job.get("mode", "batch")).lower()
    if mode not in MODES:
        raise ConfigError(f"job.mode must be one of {list(MODES)}, got {mode!r}")

    serve_raw = raw.get("serve", {})
    serve_auth_token = (
        _resolve_env_placeholders(serve_raw["auth_token"], "serve.auth_token")
        if "auth_token" in serve_raw
        else None
    )
    if mode == "serve" and not serve_auth_token:
        raise ConfigError('serve.auth_token is required when job.mode = "serve"')

    trigger_raw = job.get("trigger", {})
    trigger_enabled = bool(trigger_raw.get("enabled", False))
    trigger_auth_token = (
        _resolve_env_placeholders(trigger_raw["auth_token"], "job.trigger.auth_token")
        if "auth_token" in trigger_raw
        else None
    )
    if trigger_enabled and not trigger_auth_token:
        raise ConfigError("job.trigger.auth_token is required when job.trigger.enabled = true")

    dashboard_raw = raw.get("dashboard", {})
    dashboard_enabled = bool(dashboard_raw.get("enabled", False))

    log_epoch_metrics = bool(job.get("log_epoch_metrics", False))

    item_based = job.get("item_based", {}) or {}
    content_fallback = job.get("content_fallback", {}) or {}

    return Settings(
        input=_load_io_settings(raw, "input"),
        output=_load_io_settings(raw, "output"),
        feature_config_path=job.get("feature_config_path", "/app/config/features.toml"),
        top_k=_require_positive_int(int(job.get("top_k", 10)), name="job.top_k"),
        half_life_days=_require_positive_float(
            float(job.get("half_life_days", 90)), name="job.half_life_days"
        ),
        cron_schedule=job.get("cron_schedule", "0 3 * * *"),
        models=models,
        model_weights=model_weights,
        rrf_k=rrf_k,
        save_model_artifact=bool(job.get("save_model_artifact", False)),
        max_workers=resolve_max_workers(job.get("max_workers")),
        epoch_metrics=resolve_epoch_metrics(
            log_epoch_metrics=log_epoch_metrics,
            every=job.get("epoch_metrics_every"),
            max_users=job.get("epoch_metrics_max_users"),
            regression_drop=job.get("epoch_metrics_regression_drop"),
            plateau_eps=job.get("epoch_metrics_plateau_eps"),
            plateau_window=job.get("epoch_metrics_plateau_window"),
        ),
        item_based_k_neighbors=_require_positive_int(
            int(item_based.get("k_neighbors", DEFAULT_ITEM_BASED_K_NEIGHBORS)),
            name="job.item_based.k_neighbors",
        ),
        content_fallback_enabled=bool(content_fallback.get("enabled", False)),
        content_fallback_max_neighbors=_require_positive_int(
            int(content_fallback.get("max_neighbors", DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS)),
            name="job.content_fallback.max_neighbors",
        ),
        automl=AutomlSettings(
            enabled=bool(automl.get("enabled", False)),
            n_splits=_require_positive_int(
                int(automl.get("n_splits", AUTOML_DEFAULT_N_SPLITS)), name="job.automl.n_splits"
            ),
            test_days=_require_positive_int(
                int(automl.get("test_days", AUTOML_DEFAULT_TEST_DAYS)), name="job.automl.test_days"
            ),
            primary_metric=automl.get("primary_metric", AUTOML_DEFAULT_PRIMARY_METRIC),
            candidates=(
                [dict(candidate) for candidate in automl["candidates"]] if "candidates" in automl else None
            ),
        ),
        mode=cast(Mode, mode),
        serve=ServeSettings(
            host=serve_raw.get("host", "0.0.0.0"),
            port=int(serve_raw.get("port", 8000)),
            auth_token=serve_auth_token,
            default_k=_require_positive_int(int(serve_raw.get("default_k", 10)), name="serve.default_k"),
            refresh_interval_seconds=_require_positive_float(
                float(serve_raw.get("refresh_interval_seconds", 60)),
                name="serve.refresh_interval_seconds",
            ),
            category_column=str(serve_raw.get("category_column", "category")),
        ),
        trigger=TriggerSettings(
            enabled=trigger_enabled,
            host=trigger_raw.get("host", "0.0.0.0"),
            port=int(trigger_raw.get("port", 8080)),
            auth_token=trigger_auth_token,
            debounce_seconds=_require_positive_float(
                float(trigger_raw.get("debounce_seconds", 60)),
                name="job.trigger.debounce_seconds",
            ),
            poll_input_bucket=bool(trigger_raw.get("poll_input_bucket", False)),
            poll_interval_seconds=_require_positive_float(
                float(trigger_raw.get("poll_interval_seconds", 300)),
                name="job.trigger.poll_interval_seconds",
            ),
        ),
        dashboard=DashboardSettings(
            enabled=dashboard_enabled,
            host=dashboard_raw.get("host", "0.0.0.0"),
            port=int(dashboard_raw.get("port", 8090)),
            users_path=dashboard_raw.get("users_path", "/app/config/dashboard_users.toml"),
            refresh_interval_seconds=_require_positive_float(
                float(dashboard_raw.get("refresh_interval_seconds", 30)),
                name="dashboard.refresh_interval_seconds",
            ),
            history_limit=_require_positive_int(
                int(dashboard_raw.get("history_limit", 20)), name="dashboard.history_limit"
            ),
        ),
    )
