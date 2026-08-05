"""Configuration for the Cicerone recommender job.

Loaded from a TOML file (default /app/config/cicerone.toml, override with
CICERONE_CONFIG_PATH). Secrets use ${ENV_VAR_NAME} placeholders; escape a
literal "${...}" as "$${...}".
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class EpochMetricsSettings:
    """Tunables for optional LightFM per-epoch metric logging."""

    every: int
    max_users: int = DEFAULT_EPOCH_METRICS_MAX_USERS
    regression_drop: float = DEFAULT_EPOCH_METRICS_REGRESSION_DROP
    plateau_eps: float = DEFAULT_EPOCH_METRICS_PLATEAU_EPS
    plateau_window: int = DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW


def resolve_max_workers(raw: Any | None = None) -> int:
    """Process-pool size for AutoML folds / strategy fitting.

    Omit or pass ``None`` for sequential (``1``). An explicit integer must be >= 1.
    """
    if raw is None:
        return DEFAULT_MAX_WORKERS
    workers = int(raw)
    if workers < 1:
        raise RuntimeError(f"job.max_workers must be >= 1, got {workers}")
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
        raise ValueError(f"{context} value(s) must be non-negative, got {negative_weights}")


def validate_rrf_k(rrf_k: float | None, *, context: str = "rrf_k") -> None:
    if rrf_k is not None and rrf_k <= 0:
        raise ValueError(f"{context} must be positive, got {rrf_k}")


def _require_positive_int(value: int, *, name: str) -> int:
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1, got {value}")
    return value


def _require_positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")
    return value


def _require_unit_interval(value: float, *, name: str) -> float:
    """Require a relative fraction in (0, 1]."""
    if value <= 0 or value > 1:
        raise RuntimeError(f"{name} must be in (0, 1], got {value}")
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


MODES: tuple[str, ...] = ("batch", "serve")


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
    automl_enabled: bool
    automl_n_splits: int
    automl_test_days: int
    automl_primary_metric: str
    automl_candidates: list[dict[str, Any]] | None
    mode: str
    serve_host: str
    serve_port: int
    serve_auth_token: str | None
    serve_default_k: int
    serve_refresh_interval_seconds: float
    serve_category_column: str
    trigger_enabled: bool
    trigger_host: str
    trigger_port: int
    trigger_auth_token: str | None
    trigger_debounce_seconds: float
    trigger_poll_input_bucket: bool
    trigger_poll_interval_seconds: float
    dashboard_enabled: bool
    dashboard_host: str
    dashboard_port: int
    dashboard_users_path: str
    dashboard_refresh_interval_seconds: float
    dashboard_history_limit: int


def make_settings(**overrides: Any) -> Settings:
    """Build a fully-populated ``Settings`` with the same defaults as a typical TOML load.

    Used by tests and OpenAPI schema export so callers only pass the fields they
    care about (``mode``, auth tokens, …) and stay aligned as ``Settings`` grows.
    Prefer ``load_settings(path)`` for real config files.
    """
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
        automl_enabled=False,
        automl_n_splits=AUTOML_DEFAULT_N_SPLITS,
        automl_test_days=AUTOML_DEFAULT_TEST_DAYS,
        automl_primary_metric=AUTOML_DEFAULT_PRIMARY_METRIC,
        automl_candidates=None,
        mode="batch",
        serve_host="0.0.0.0",
        serve_port=8000,
        serve_auth_token=None,
        serve_default_k=10,
        serve_refresh_interval_seconds=60,
        serve_category_column="category",
        trigger_enabled=False,
        trigger_host="0.0.0.0",
        trigger_port=8080,
        trigger_auth_token=None,
        trigger_debounce_seconds=60,
        trigger_poll_input_bucket=False,
        trigger_poll_interval_seconds=300,
        dashboard_enabled=False,
        dashboard_host="0.0.0.0",
        dashboard_port=8090,
        dashboard_users_path="/tmp/dashboard_users.toml",
        dashboard_refresh_interval_seconds=30,
        dashboard_history_limit=20,
    )
    base.update(overrides)
    return Settings(**base)


def _load_io_settings(raw: dict[str, Any], section_name: str) -> IOSettings:
    section = raw.get(section_name)
    if not section:
        raise RuntimeError(f"Missing required config section: [{section_name}]")
    if "kind" not in section:
        raise RuntimeError(f"Missing required config key: [{section_name}].kind")
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
            raise RuntimeError(
                "job.models is empty; configure at least one model name, or omit job.models entirely "
                "to use the default"
            )
        unknown_models = [name for name in models if name not in STRATEGY_NAMES]
        if unknown_models:
            raise RuntimeError(
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
            raise RuntimeError(f"job.model_weights key(s) {unknown_weights} are not in job.models {models}")
    rrf_k = float(job["rrf_k"]) if "rrf_k" in job else None
    validate_rrf_k(rrf_k, context="job.rrf_k")

    mode = str(job.get("mode", "batch")).lower()
    if mode not in MODES:
        raise RuntimeError(f"job.mode must be one of {list(MODES)}, got {mode!r}")

    serve = raw.get("serve", {})
    serve_auth_token = (
        _resolve_env_placeholders(serve["auth_token"], "serve.auth_token") if "auth_token" in serve else None
    )
    if mode == "serve" and not serve_auth_token:
        raise RuntimeError('serve.auth_token is required when job.mode = "serve"')

    trigger = job.get("trigger", {})
    trigger_enabled = bool(trigger.get("enabled", False))
    trigger_auth_token = (
        _resolve_env_placeholders(trigger["auth_token"], "job.trigger.auth_token")
        if "auth_token" in trigger
        else None
    )
    if trigger_enabled and not trigger_auth_token:
        raise RuntimeError("job.trigger.auth_token is required when job.trigger.enabled = true")

    dashboard = raw.get("dashboard", {})
    dashboard_enabled = bool(dashboard.get("enabled", False))

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
        automl_enabled=bool(automl.get("enabled", False)),
        automl_n_splits=_require_positive_int(
            int(automl.get("n_splits", AUTOML_DEFAULT_N_SPLITS)), name="job.automl.n_splits"
        ),
        automl_test_days=_require_positive_int(
            int(automl.get("test_days", AUTOML_DEFAULT_TEST_DAYS)), name="job.automl.test_days"
        ),
        automl_primary_metric=automl.get("primary_metric", AUTOML_DEFAULT_PRIMARY_METRIC),
        automl_candidates=(
            [dict(candidate) for candidate in automl["candidates"]] if "candidates" in automl else None
        ),
        mode=mode,
        serve_host=serve.get("host", "0.0.0.0"),
        serve_port=int(serve.get("port", 8000)),
        serve_auth_token=serve_auth_token,
        serve_default_k=_require_positive_int(int(serve.get("default_k", 10)), name="serve.default_k"),
        serve_refresh_interval_seconds=float(serve.get("refresh_interval_seconds", 60)),
        serve_category_column=str(serve.get("category_column", "category")),
        trigger_enabled=trigger_enabled,
        trigger_host=trigger.get("host", "0.0.0.0"),
        trigger_port=int(trigger.get("port", 8080)),
        trigger_auth_token=trigger_auth_token,
        trigger_debounce_seconds=float(trigger.get("debounce_seconds", 60)),
        trigger_poll_input_bucket=bool(trigger.get("poll_input_bucket", False)),
        trigger_poll_interval_seconds=float(trigger.get("poll_interval_seconds", 300)),
        dashboard_enabled=dashboard_enabled,
        dashboard_host=dashboard.get("host", "0.0.0.0"),
        dashboard_port=int(dashboard.get("port", 8090)),
        dashboard_users_path=dashboard.get("users_path", "/app/config/dashboard_users.toml"),
        dashboard_refresh_interval_seconds=float(dashboard.get("refresh_interval_seconds", 30)),
        dashboard_history_limit=_require_positive_int(
            int(dashboard.get("history_limit", 20)), name="dashboard.history_limit"
        ),
    )
