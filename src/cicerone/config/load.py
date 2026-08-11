"""Load Settings from TOML / build Settings for tests."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from cicerone.config.constants import (
    AUTOML_DEFAULT_N_SPLITS,
    AUTOML_DEFAULT_PRIMARY_METRIC,
    AUTOML_DEFAULT_TEST_DAYS,
    DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    DEFAULT_ITEM_BASED_K_NEIGHBORS,
    DEFAULT_LOCK_KEY,
    DEFAULT_LOCK_TTL_SECONDS,
    DEFAULT_MAX_WORKERS,
    LOCK_BACKENDS,
    MODES,
    STRATEGY_NAMES,
    ConfigError,
    Mode,
)
from cicerone.config.settings import (
    AutomlSettings,
    DashboardSettings,
    IOSettings,
    ServeSettings,
    Settings,
    TriggerSettings,
)
from cicerone.config.validation import (
    require_positive_float,
    require_positive_int,
    resolve_epoch_metrics,
    resolve_max_workers,
    validate_model_weights,
    validate_rrf_k,
)

_ENV_PLACEHOLDER = re.compile(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)\}")

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
    ("trigger_lock_backend", "lock_backend"),
    ("trigger_postgres_url", "postgres_url"),
    ("trigger_redis_url", "redis_url"),
    ("trigger_lock_key", "lock_key"),
    ("trigger_lock_ttl_seconds", "lock_ttl_seconds"),
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
    """``Settings`` with TOML-like defaults; overrides for tests / OpenAPI export."""
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
        model_configs=None,
        content_fallback_enabled=False,
        content_fallback_max_neighbors=DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
        automl=automl,
        mode="batch",
        serve=serve,
        trigger=trigger,
        dashboard=dashboard,
    )
    base.update(overrides)
    if base.get("model_configs") is None:
        from cicerone.model_config import item_based_k_from_config, resolve_model_configs

        k_neighbors = int(base["item_based_k_neighbors"])
        k_explicit = "item_based_k_neighbors" in overrides
        configs = resolve_model_configs(
            legacy_k_neighbors=k_neighbors,
            legacy_k_neighbors_explicit=k_explicit,
        )
        base["model_configs"] = configs
        k_from_cfg = item_based_k_from_config(configs["item_based"])
        if k_from_cfg is not None:
            base["item_based_k_neighbors"] = k_from_cfg
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
    # Read DEFAULT_CONFIG_PATH via the package so tests can monkeypatch
    # ``cicerone.config.DEFAULT_CONFIG_PATH``.
    import cicerone.config as config_pkg

    path = Path(config_path or os.environ.get("CICERONE_CONFIG_PATH") or config_pkg.DEFAULT_CONFIG_PATH)
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

    lock_backend = str(trigger_raw.get("lock_backend", "in_process")).lower()
    if lock_backend not in LOCK_BACKENDS:
        raise ConfigError(
            f"job.trigger.lock_backend must be one of {list(LOCK_BACKENDS)}, got {lock_backend!r}"
        )
    trigger_postgres_url = (
        _resolve_env_placeholders(trigger_raw["postgres_url"], "job.trigger.postgres_url")
        if "postgres_url" in trigger_raw
        else None
    )
    trigger_redis_url = (
        _resolve_env_placeholders(trigger_raw["redis_url"], "job.trigger.redis_url")
        if "redis_url" in trigger_raw
        else None
    )
    if lock_backend == "redis" and not trigger_redis_url:
        raise ConfigError('job.trigger.redis_url is required when lock_backend = "redis"')

    dashboard_raw = raw.get("dashboard", {})
    dashboard_enabled = bool(dashboard_raw.get("enabled", False))

    log_epoch_metrics = bool(job.get("log_epoch_metrics", False))

    item_based = job.get("item_based", {}) or {}
    content_fallback = job.get("content_fallback", {}) or {}
    legacy_k_explicit = "k_neighbors" in item_based
    legacy_k_neighbors = require_positive_int(
        int(item_based.get("k_neighbors", DEFAULT_ITEM_BASED_K_NEIGHBORS)),
        name="job.item_based.k_neighbors",
    )

    from cicerone.model_config import item_based_k_from_config, resolve_model_configs

    model_configs = resolve_model_configs(
        raw.get("model"),
        legacy_k_neighbors=legacy_k_neighbors,
        legacy_k_neighbors_explicit=legacy_k_explicit,
    )
    resolved_k = item_based_k_from_config(model_configs["item_based"])
    item_based_k_neighbors = (
        require_positive_int(resolved_k, name="model.item_based.model.K")
        if resolved_k is not None
        else legacy_k_neighbors
    )

    input_settings = _load_io_settings(raw, "input")
    output_settings = _load_io_settings(raw, "output")
    if lock_backend == "postgres":
        from cicerone.config.lock_url import require_postgres_lock_url_parts

        require_postgres_lock_url_parts(
            postgres_url=trigger_postgres_url,
            output_kind=output_settings.kind,
            output_options=output_settings.options,
        )

    trigger_lock_key = str(trigger_raw.get("lock_key", DEFAULT_LOCK_KEY)).strip()
    if not trigger_lock_key:
        raise ConfigError("job.trigger.lock_key must be a non-empty string")
    trigger_lock_ttl_seconds = require_positive_float(
        float(trigger_raw.get("lock_ttl_seconds", DEFAULT_LOCK_TTL_SECONDS)),
        name="job.trigger.lock_ttl_seconds",
    )

    return Settings(
        input=input_settings,
        output=output_settings,
        feature_config_path=job.get("feature_config_path", "/app/config/features.toml"),
        top_k=require_positive_int(int(job.get("top_k", 10)), name="job.top_k"),
        half_life_days=require_positive_float(
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
        item_based_k_neighbors=item_based_k_neighbors,
        model_configs=model_configs,
        content_fallback_enabled=bool(content_fallback.get("enabled", False)),
        content_fallback_max_neighbors=require_positive_int(
            int(content_fallback.get("max_neighbors", DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS)),
            name="job.content_fallback.max_neighbors",
        ),
        automl=AutomlSettings(
            enabled=bool(automl.get("enabled", False)),
            n_splits=require_positive_int(
                int(automl.get("n_splits", AUTOML_DEFAULT_N_SPLITS)), name="job.automl.n_splits"
            ),
            test_days=require_positive_int(
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
            default_k=require_positive_int(int(serve_raw.get("default_k", 10)), name="serve.default_k"),
            refresh_interval_seconds=require_positive_float(
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
            debounce_seconds=require_positive_float(
                float(trigger_raw.get("debounce_seconds", 60)),
                name="job.trigger.debounce_seconds",
            ),
            poll_input_bucket=bool(trigger_raw.get("poll_input_bucket", False)),
            poll_interval_seconds=require_positive_float(
                float(trigger_raw.get("poll_interval_seconds", 300)),
                name="job.trigger.poll_interval_seconds",
            ),
            lock_backend=lock_backend,
            postgres_url=trigger_postgres_url,
            redis_url=trigger_redis_url,
            lock_key=trigger_lock_key,
            lock_ttl_seconds=trigger_lock_ttl_seconds,
        ),
        dashboard=DashboardSettings(
            enabled=dashboard_enabled,
            host=dashboard_raw.get("host", "0.0.0.0"),
            port=int(dashboard_raw.get("port", 8090)),
            users_path=dashboard_raw.get("users_path", "/app/config/dashboard_users.toml"),
            refresh_interval_seconds=require_positive_float(
                float(dashboard_raw.get("refresh_interval_seconds", 30)),
                name="dashboard.refresh_interval_seconds",
            ),
            history_limit=require_positive_int(
                int(dashboard_raw.get("history_limit", 20)), name="dashboard.history_limit"
            ),
        ),
    )
