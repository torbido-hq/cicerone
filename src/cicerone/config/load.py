"""Load Settings from TOML / build Settings for tests."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from cicerone.config.constants import (
    ALLOCATION_THOMPSON,
    ARTIFACT_HMAC_KEY_MIN_BYTES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    DEFAULT_CRON_SCHEDULE,
    DEFAULT_FEATURE_CONFIG_PATH,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_ITEM_BASED_K_NEIGHBORS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MODE,
    DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS,
    DEFAULT_SERVE_MAX_K,
    DEFAULT_TOP_K,
    LOCK_BACKENDS,
    MODES,
    PRIMARY_METRIC_CONVERSION,
    PRIMARY_METRIC_CTR,
    STRATEGY_NAMES,
    ConfigError,
    Mode,
)
from cicerone.config.events import load_events_settings
from cicerone.config.experiment import coerce_experiment, load_experiment_settings
from cicerone.config.publish import load_publish_settings
from cicerone.config.settings import (
    AutomlSettings,
    DashboardSettings,
    ExplainSettings,
    IOSettings,
    ServeSettings,
    Settings,
    TriggerSettings,
)
from cicerone.config.track import load_eval_settings, load_track_settings
from cicerone.config.unknown_keys import warn_unknown_keys
from cicerone.config.validation import (
    float_field,
    int_field,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
    resolve_env_placeholders,
    resolve_epoch_metrics,
    validate_model_weights,
    validate_rrf_k,
)

logger = logging.getLogger(__name__)

_MAKE_SETTINGS_IO = {
    "input": {"kind": "dataset", "options": {"storage_backend": "local", "path": "/tmp/in"}},
    "output": {"kind": "dataset", "options": {"storage_backend": "local", "path": "/tmp/out"}},
}
_SECTIONS = (
    "input",
    "output",
    "serve",
    "trigger",
    "dashboard",
    "automl",
    "explain",
    "events",
    "publish",
    "track",
    "eval",
)


def _optional_env(raw: dict[str, Any], key: str, path: str) -> Any:
    return resolve_env_placeholders(raw[key], path) if key in raw else None


def _serve_default_k(serve_raw: dict[str, Any]) -> int:
    value = int_field(serve_raw, "serve.default_k", ServeSettings.default_k, require_positive_int)
    if value > DEFAULT_SERVE_MAX_K:
        raise ConfigError(f"serve.default_k must be <= {DEFAULT_SERVE_MAX_K}, got {value}")
    return value


def _load_serve_settings(serve_raw: dict[str, Any], mode: str) -> ServeSettings:
    auth_token = _optional_env(serve_raw, "auth_token", "serve.auth_token")
    if mode == "serve" and not auth_token:
        raise ConfigError('serve.auth_token is required when job.mode = "serve"')
    metrics_token = (
        resolve_env_placeholders(serve_raw["metrics_token"], "serve.metrics_token")
        if serve_raw.get("metrics_token")
        else None
    )
    if isinstance(metrics_token, str):
        metrics_token = metrics_token.strip() or None
    metrics_enabled = bool(serve_raw.get("metrics_enabled", ServeSettings.metrics_enabled))
    if metrics_enabled and not metrics_token:
        raise ConfigError("serve.metrics_token is required when serve.metrics_enabled = true")
    return ServeSettings(
        host=serve_raw.get("host", ServeSettings.host),
        port=int(serve_raw.get("port", ServeSettings.port)),
        auth_token=auth_token,
        default_k=_serve_default_k(serve_raw),
        refresh_interval_seconds=float_field(
            serve_raw,
            "serve.refresh_interval_seconds",
            ServeSettings.refresh_interval_seconds,
            require_positive_float,
        ),
        category_column=str(serve_raw.get("category_column", ServeSettings.category_column)),
        metrics_enabled=metrics_enabled,
        metrics_token=metrics_token,
        log_impressions=bool(serve_raw.get("log_impressions", ServeSettings.log_impressions)),
    )


def _load_trigger_settings(trigger_raw: dict[str, Any]) -> TriggerSettings:
    enabled = bool(trigger_raw.get("enabled", TriggerSettings.enabled))
    auth_token = _optional_env(trigger_raw, "auth_token", "job.trigger.auth_token")
    if enabled and not auth_token:
        raise ConfigError("job.trigger.auth_token is required when job.trigger.enabled = true")
    lock_backend = str(trigger_raw.get("lock_backend", TriggerSettings.lock_backend)).lower()
    if lock_backend not in LOCK_BACKENDS:
        raise ConfigError(
            f"job.trigger.lock_backend must be one of {list(LOCK_BACKENDS)}, got {lock_backend!r}"
        )
    postgres_url = _optional_env(trigger_raw, "postgres_url", "job.trigger.postgres_url")
    redis_url = _optional_env(trigger_raw, "redis_url", "job.trigger.redis_url")
    if lock_backend == "redis" and not redis_url:
        raise ConfigError('job.trigger.redis_url is required when lock_backend = "redis"')
    lock_key = str(trigger_raw.get("lock_key", TriggerSettings.lock_key)).strip()
    if not lock_key:
        raise ConfigError("job.trigger.lock_key must be a non-empty string")
    return TriggerSettings(
        enabled=enabled,
        host=trigger_raw.get("host", TriggerSettings.host),
        port=int(trigger_raw.get("port", TriggerSettings.port)),
        auth_token=auth_token,
        debounce_seconds=float_field(
            trigger_raw,
            "job.trigger.debounce_seconds",
            TriggerSettings.debounce_seconds,
            require_positive_float,
        ),
        poll_input_bucket=bool(trigger_raw.get("poll_input_bucket", TriggerSettings.poll_input_bucket)),
        poll_interval_seconds=float_field(
            trigger_raw,
            "job.trigger.poll_interval_seconds",
            TriggerSettings.poll_interval_seconds,
            require_positive_float,
        ),
        lock_backend=lock_backend,
        postgres_url=postgres_url,
        redis_url=redis_url,
        lock_key=lock_key,
        lock_ttl_seconds=float_field(
            trigger_raw,
            "job.trigger.lock_ttl_seconds",
            TriggerSettings.lock_ttl_seconds,
            require_positive_float,
        ),
    )


def _load_lookup_user_attrs(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple) or not all(isinstance(item, str) for item in raw):
        raise ConfigError("dashboard.lookup_user_attrs must be a list of strings")
    return tuple(dict.fromkeys(item.strip() for item in raw if item.strip() and item.strip() != "user_id"))


def _load_dashboard_settings(dashboard_raw: dict[str, Any]) -> DashboardSettings:
    return DashboardSettings(
        enabled=bool(dashboard_raw.get("enabled", DashboardSettings.enabled)),
        host=dashboard_raw.get("host", DashboardSettings.host),
        port=int(dashboard_raw.get("port", DashboardSettings.port)),
        users_path=dashboard_raw.get("users_path", DashboardSettings.users_path),
        refresh_interval_seconds=float_field(
            dashboard_raw,
            "dashboard.refresh_interval_seconds",
            DashboardSettings.refresh_interval_seconds,
            require_positive_float,
        ),
        history_limit=int_field(
            dashboard_raw, "dashboard.history_limit", DashboardSettings.history_limit, require_positive_int
        ),
        lookup_k=int_field(
            dashboard_raw, "dashboard.lookup_k", DashboardSettings.lookup_k, require_positive_int
        ),
        lookup_events=int_field(
            dashboard_raw, "dashboard.lookup_events", DashboardSettings.lookup_events, require_positive_int
        ),
        lookup_user_attrs=_load_lookup_user_attrs(dashboard_raw.get("lookup_user_attrs")),
    )


def _load_automl_settings(automl: dict[str, Any]) -> AutomlSettings:
    candidates = automl.get("candidates")
    return AutomlSettings(
        enabled=bool(automl.get("enabled", AutomlSettings.enabled)),
        n_splits=int_field(automl, "job.automl.n_splits", AutomlSettings.n_splits, require_positive_int),
        test_days=int_field(automl, "job.automl.test_days", AutomlSettings.test_days, require_positive_int),
        primary_metric=automl.get("primary_metric", AutomlSettings.primary_metric),
        candidates=[dict(candidate) for candidate in candidates] if candidates is not None else None,
        debias=bool(automl.get("debias", AutomlSettings.debias)),
    )


def _load_explain_settings(raw: dict[str, Any]) -> ExplainSettings:
    return ExplainSettings(
        enabled=bool(raw.get("enabled", ExplainSettings.enabled)),
        max_similar_items=int_field(
            raw, "job.explain.max_similar_items", ExplainSettings.max_similar_items, require_non_negative_int
        ),
        max_attributes=int_field(
            raw, "job.explain.max_attributes", ExplainSettings.max_attributes, require_non_negative_int
        ),
    )


def _load_io_settings(section: dict[str, Any] | None, section_name: str) -> IOSettings:
    if not section:
        raise ConfigError(f"Missing required config section: [{section_name}]")
    if "kind" not in section:
        raise ConfigError(f"Missing required config key: [{section_name}].kind")
    options = resolve_env_placeholders(section.get("options", {}), f"{section_name}.options")
    hmac_key = section.get("artifact_hmac_key")
    if hmac_key is not None:
        if section_name != "output":
            raise ConfigError("artifact_hmac_key is only valid on [output]")
        if not isinstance(hmac_key, str):
            raise ConfigError("output.artifact_hmac_key must be a string")
        hmac_key = resolve_env_placeholders(hmac_key, "output.artifact_hmac_key").strip() or None
        if hmac_key is not None and len(hmac_key.encode("utf-8")) < ARTIFACT_HMAC_KEY_MIN_BYTES:
            raise ConfigError(
                f"output.artifact_hmac_key must be at least {ARTIFACT_HMAC_KEY_MIN_BYTES} bytes"
            )
    return IOSettings(kind=str(section["kind"]).lower(), options=options, artifact_hmac_key=hmac_key)


def _load_sections(sections: dict[str, Any], mode: str) -> dict[str, Any]:
    """Settings sections from TOML-shaped tables, keyed by ``Settings`` field name."""
    serve = _load_serve_settings(sections["serve"], mode)
    return {
        "input": _load_io_settings(sections["input"], "input"),
        "output": _load_io_settings(sections["output"], "output"),
        "serve": serve,
        "trigger": _load_trigger_settings(sections["trigger"]),
        "dashboard": _load_dashboard_settings(sections["dashboard"]),
        "automl": _load_automl_settings(sections["automl"]),
        "explain": _load_explain_settings(sections["explain"]),
        "events": load_events_settings(sections["events"], mode=mode, serve_auth_token=serve.auth_token),
        "publish": load_publish_settings(sections["publish"]),
        "track": load_track_settings(sections["track"]),
        "eval": load_eval_settings(sections["eval"]),
    }


def _section_raw(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return value if isinstance(value, dict) else asdict(value)


def make_settings(**overrides: Any) -> Settings:
    """``Settings`` with TOML-like defaults; overrides for tests / OpenAPI export."""
    mode = overrides.pop("mode", DEFAULT_MODE)
    sections = {name: _section_raw(overrides.pop(name, _MAKE_SETTINGS_IO.get(name))) for name in _SECTIONS}
    base: dict[str, Any] = dict(
        feature_config_path=DEFAULT_FEATURE_CONFIG_PATH,
        top_k=DEFAULT_TOP_K,
        half_life_days=DEFAULT_HALF_LIFE_DAYS,
        cron_schedule=DEFAULT_CRON_SCHEDULE,
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
        sequential_min_median_interactions=DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS,
        mode=mode,
        experiment=coerce_experiment(overrides.pop("experiment", None)),
        **_load_sections(sections, mode),
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
    return _finalize_settings(Settings(**base))


def _finalize_settings(settings: Settings) -> Settings:
    _require_lock_backend(settings)
    _require_online_artifact_hmac(settings)
    _require_exposure_log_backend(settings)
    _require_online_output_backend(settings)
    _require_online_collaborative_lightfm(settings)
    _require_track_backend(settings)
    _require_thompson_allocation(settings)
    _warn_online_skipped_for_experiment(settings)
    _warn_online_skipped_for_sequential(settings)
    return settings


def _require_lock_backend(settings: Settings) -> None:
    trigger = settings.trigger
    if trigger.lock_backend == "postgres":
        from cicerone.config.lock_url import require_postgres_lock_url_parts

        require_postgres_lock_url_parts(
            postgres_url=trigger.postgres_url,
            output_kind=settings.output.kind,
            output_options=settings.output.options,
        )
    if settings.events.enabled and settings.events.ha and trigger.lock_backend == "in_process":
        raise ConfigError(
            'events.ha = true requires job.trigger.lock_backend "postgres" or "redis" '
            "(leader-only incremental apply)"
        )


def _require_exposure_log_backend(settings: Settings) -> None:
    if not settings.experiment.enabled or not settings.experiment.log_exposures:
        return
    from cicerone.experiment.store import EXPOSURE_LOG_HA_ERROR, require_appendable_exposure_log

    require_appendable_exposure_log(settings.output)
    if settings.events.ha and settings.output.kind != "db":
        raise ConfigError(EXPOSURE_LOG_HA_ERROR)


ONLINE_OUTPUT_ERROR = (
    'events.online.enabled requires output kind = "db" or a local dataset path; '
    "S3 object replace is not compare-and-swap"
)
ONLINE_HMAC_ERROR = "events.online.enabled requires output.artifact_hmac_key"


def _require_online_artifact_hmac(settings: Settings) -> None:
    if settings.events.online.enabled and not settings.output.artifact_hmac_key:
        raise ConfigError(ONLINE_HMAC_ERROR)


def _require_online_collaborative_lightfm(settings: Settings) -> None:
    if not settings.events.online.enabled:
        return
    from cicerone.model_config import LIGHTFM_WRAPPER_CLS, collaborative_cls

    cls = collaborative_cls(settings.model_configs.get("collaborative"))
    if cls != LIGHTFM_WRAPPER_CLS:
        raise ConfigError(
            "events.online.enabled requires [model.collaborative].cls = "
            f"{LIGHTFM_WRAPPER_CLS!r}; got {cls!r}. Use ease or als as their own "
            "job.models names instead of swapping the collaborative slot"
        )


def _require_online_output_backend(settings: Settings) -> None:
    if not settings.events.online.enabled:
        return
    from cicerone.io.options import storage_backend

    output = settings.output
    if output.kind == "db":
        return
    if output.kind == "dataset" and storage_backend(output.options) == "local":
        return
    raise ConfigError(ONLINE_OUTPUT_ERROR)


def _require_track_backend(settings: Settings) -> None:
    if settings.serve.log_impressions and not settings.track.enabled:
        raise ConfigError("serve.log_impressions requires track.enabled = true")
    rec_metric = settings.experiment.primary_metric in {PRIMARY_METRIC_CTR, PRIMARY_METRIC_CONVERSION}
    if settings.experiment.enabled and rec_metric and not settings.track.enabled:
        raise ConfigError("experiment.primary_metric 'ctr'/'conversion' requires track.enabled = true")
    if not settings.track.enabled:
        return
    from cicerone.track.store import TRACK_LOG_HA_ERROR, require_appendable_track_log

    require_appendable_track_log(settings.output)
    if settings.events.ha and settings.output.kind != "db":
        raise ConfigError(TRACK_LOG_HA_ERROR)


def _require_thompson_allocation(settings: Settings) -> None:
    if not settings.experiment.enabled or settings.experiment.allocation != ALLOCATION_THOMPSON:
        return
    from cicerone.experiment.thompson import require_bandits_extra

    require_bandits_extra()
    if not settings.track.enabled:
        raise ConfigError("experiment.allocation 'thompson' requires track.enabled = true")


def _warn_online_skipped_for_experiment(settings: Settings) -> None:
    if settings.events.online.enabled and settings.experiment.enabled:
        logger.warning(
            "[events.online] is enabled while [experiment] is enabled; "
            "online collaborative refresh will be skipped"
        )


def _warn_online_skipped_for_sequential(settings: Settings) -> None:
    if not settings.events.online.enabled:
        return
    names = settings.models
    if names is None:
        from cicerone.config.constants import DEFAULT_MODELS

        names = DEFAULT_MODELS
    if "sequential" not in names:
        return
    from cicerone.model_config import sequential_extra_available

    if sequential_extra_available():
        return
    logger.warning(
        "[events.online] is enabled while job.models includes sequential "
        "and the torch extra is not installed; online recommend will be skipped"
    )


def load_settings(config_path: str | None = None) -> Settings:
    path = Path(config_path or os.environ.get("CICERONE_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    warn_unknown_keys(raw)
    job = raw.get("job", {})
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

    mode = str(job.get("mode", DEFAULT_MODE)).lower()
    if mode not in MODES:
        raise ConfigError(f"job.mode must be one of {list(MODES)}, got {mode!r}")

    sections = _load_sections(
        {
            "input": raw.get("input"),
            "output": raw.get("output"),
            "serve": raw.get("serve", {}),
            "trigger": job.get("trigger", {}),
            "dashboard": raw.get("dashboard", {}),
            "automl": job.get("automl", {}),
            "explain": job.get("explain", {}) or {},
            "events": raw.get("events", {}) or {},
            "publish": raw.get("publish", {}) or {},
            "track": raw.get("track") or {},
            "eval": job.get("eval") or {},
        },
        mode,
    )

    item_based = job.get("item_based", {}) or {}
    content_fallback = job.get("content_fallback", {}) or {}
    sequential = job.get("sequential", {}) or {}
    legacy_k_explicit = "k_neighbors" in item_based
    legacy_k_neighbors = int_field(
        item_based, "job.item_based.k_neighbors", DEFAULT_ITEM_BASED_K_NEIGHBORS, require_positive_int
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

    settings = Settings(
        feature_config_path=job.get("feature_config_path", DEFAULT_FEATURE_CONFIG_PATH),
        top_k=int_field(job, "job.top_k", DEFAULT_TOP_K, require_positive_int),
        half_life_days=float_field(job, "job.half_life_days", DEFAULT_HALF_LIFE_DAYS, require_positive_float),
        cron_schedule=job.get("cron_schedule", DEFAULT_CRON_SCHEDULE),
        models=models,
        model_weights=model_weights,
        rrf_k=rrf_k,
        save_model_artifact=bool(job.get("save_model_artifact", False)),
        max_workers=int_field(job, "job.max_workers", DEFAULT_MAX_WORKERS, require_positive_int),
        epoch_metrics=resolve_epoch_metrics(job),
        item_based_k_neighbors=item_based_k_neighbors,
        model_configs=model_configs,
        content_fallback_enabled=bool(content_fallback.get("enabled", False)),
        content_fallback_max_neighbors=int_field(
            content_fallback,
            "job.content_fallback.max_neighbors",
            DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
            require_positive_int,
        ),
        sequential_min_median_interactions=int_field(
            sequential,
            "job.sequential.min_median_interactions",
            DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS,
            require_positive_int,
        ),
        mode=cast(Mode, mode),
        experiment=load_experiment_settings(raw.get("experiment") or {}),
        **sections,
    )
    return _finalize_settings(settings)
