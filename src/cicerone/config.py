"""Configuration for the Cicerone recommender job.

Loaded from a single TOML file (default /app/config/cicerone.toml, override
with CICERONE_CONFIG_PATH). Secrets are never stored in the file itself:
reference them with ${ENV_VAR_NAME} placeholders, resolved from the process
environment at load time. Escape a literal "${...}" by doubling the leading
"$", e.g. "$${LITERAL}".

Input/output backends are each configured generically as a "kind" plus a
free-form "options" table interpreted by the corresponding module in
cicerone.io — see cicerone.io.factory.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "/app/config/cicerone.toml"

# Kept here (not cicerone.model, which has heavy ML deps config.py avoids
# importing) so Settings.models can be validated at load time and can't
# drift from cicerone.model.STRATEGIES.
STRATEGY_NAMES: tuple[str, ...] = ("collaborative", "item_based", "popular", "latest")

AUTOML_DEFAULT_N_SPLITS = 2
AUTOML_DEFAULT_TEST_DAYS = 14
AUTOML_DEFAULT_PRIMARY_METRIC = "MAP"


def validate_model_weights(weights: dict[str, float] | None, *, context: str = "model_weights") -> None:
    """Raises ValueError if any weight is negative."""
    if weights is None:
        return
    negative_weights = {name: weight for name, weight in weights.items() if weight < 0}
    if negative_weights:
        raise ValueError(f"{context} value(s) must be non-negative, got {negative_weights}")


def validate_rrf_k(rrf_k: float | None, *, context: str = "rrf_k") -> None:
    """Raises ValueError if rrf_k is set but not positive."""
    if rrf_k is not None and rrf_k <= 0:
        raise ValueError(f"{context} must be positive, got {rrf_k}")


_ENV_PLACEHOLDER = re.compile(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_placeholders(value: Any, path: str = "") -> Any:
    """Recursively replaces "${VAR_NAME}" with the matching environment
    variable. `path` identifies the config location in error messages.
    """
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
    """Generic I/O configuration: a backend "kind" plus its own options dict,
    interpreted by the corresponding module in cicerone.io. `kind` is
    normalized to lower case when loaded from TOML.
    """

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

    return Settings(
        input=_load_io_settings(raw, "input"),
        output=_load_io_settings(raw, "output"),
        feature_config_path=job.get("feature_config_path", "/app/config/features.toml"),
        top_k=int(job.get("top_k", 10)),
        half_life_days=float(job.get("half_life_days", 90)),
        cron_schedule=job.get("cron_schedule", "0 3 * * *"),
        models=models,
        model_weights=model_weights,
        rrf_k=rrf_k,
        automl_enabled=bool(automl.get("enabled", False)),
        automl_n_splits=int(automl.get("n_splits", AUTOML_DEFAULT_N_SPLITS)),
        automl_test_days=int(automl.get("test_days", AUTOML_DEFAULT_TEST_DAYS)),
        automl_primary_metric=automl.get("primary_metric", AUTOML_DEFAULT_PRIMARY_METRIC),
        automl_candidates=(
            [dict(candidate) for candidate in automl["candidates"]] if "candidates" in automl else None
        ),
        mode=mode,
        serve_host=serve.get("host", "0.0.0.0"),
        serve_port=int(serve.get("port", 8000)),
        serve_auth_token=serve_auth_token,
        serve_default_k=int(serve.get("default_k", 10)),
        serve_refresh_interval_seconds=float(serve.get("refresh_interval_seconds", 60)),
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
        dashboard_history_limit=int(dashboard.get("history_limit", 20)),
    )
