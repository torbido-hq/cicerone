"""Configuration for the Cicerone recommender job.

Everything is loaded from a single TOML file (default:
/app/config/cicerone.toml, override with CICERONE_CONFIG_PATH). Secrets are
never stored in the TOML file itself: reference them with ${ENV_VAR_NAME}
placeholders, resolved from the process environment at load time (see
.env.example). This keeps the structural configuration (which backend, which
bucket/table, scheduling, tuning...) in version control while credentials
stay in environment variables / secret stores.

Every "${VAR_NAME}" occurrence is resolved, including partial ones embedded
in a larger string (e.g. `prefix = "datasets/${ENV}/latest"`), and it is
mandatory: a referenced variable that isn't set raises an error rather than
silently leaving the placeholder in place. If you need a literal "${...}" in
a value (no substitution), escape it by doubling the leading "$", e.g.
`pattern = "$${LITERAL}"` resolves to the literal string "${LITERAL}".

Input and output are each independently configurable, and are deliberately
generic: a "kind" (e.g. "dataset", "db") plus a free-form "options" table
interpreted by the corresponding backend in cicerone.io. This is what makes
Cicerone adaptable to any product catalog, not tied to one particular
application: adding a new backend (a message queue, a different warehouse,
...) never requires changing this module — see cicerone.io.factory.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "/app/config/cicerone.toml"

# Canonical multi-model strategy identifiers. Centralized here (not in
# cicerone.model, which owns the actual per-strategy factories/behavior but
# has heavy ML deps -- lightfm/implicit/rectools -- that config.py
# deliberately doesn't import) so Settings.models can be validated at config
# load time, and so cicerone.model.STRATEGIES' keys, DEFAULT_MODELS, config
# file comments, and README can't silently drift out of sync with this list.
STRATEGY_NAMES: tuple[str, ...] = ("collaborative", "item_based", "popular", "latest")

# Centralized here (not in cicerone.automl, which config.py deliberately
# doesn't import) so the [job.automl] TOML defaults and automl.py's function
# defaults can't drift apart.
AUTOML_DEFAULT_N_SPLITS = 2
AUTOML_DEFAULT_TEST_DAYS = 14
AUTOML_DEFAULT_PRIMARY_METRIC = "MAP"


def validate_model_weights(weights: dict[str, float] | None, *, context: str = "model_weights") -> None:
    """Raises ValueError if any weight is negative. Shared by config.load_settings,
    model.train_and_recommend, and automl's candidate parsing so all three fail on
    the same invalid configurations with the same error shape (`context` only
    changes the message prefix so each caller's error reads naturally).
    """
    if weights is None:
        return
    negative_weights = {name: weight for name, weight in weights.items() if weight < 0}
    if negative_weights:
        raise ValueError(f"{context} value(s) must be non-negative, got {negative_weights}")


def validate_rrf_k(rrf_k: float | None, *, context: str = "rrf_k") -> None:
    """Raises ValueError if rrf_k is set but not positive. Shared by
    config.load_settings, model.train_and_recommend, and automl's candidate
    parsing (see validate_model_weights).
    """
    if rrf_k is not None and rrf_k <= 0:
        raise ValueError(f"{context} must be positive, got {rrf_k}")


_ENV_PLACEHOLDER = re.compile(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_placeholders(value: Any, path: str = "") -> Any:
    """Recursively replaces "${VAR_NAME}" occurrences with the matching
    environment variable. Supports partial interpolation (e.g.
    "datasets/${ENV}/latest") and a "$${VAR_NAME}" escape for a literal
    "${VAR_NAME}" that should not be substituted. `path` is the config
    location this value came from, included in the error message if a
    referenced environment variable is missing.
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
    """Generic I/O configuration: a backend "kind" plus its own options.

    Deliberately untyped (``options`` is a plain dict) so new input/output
    backends can be added under cicerone.io without ever touching this
    module — see cicerone.io.factory.build_input_source/build_output_sink.
    ``kind`` is normalized to lower case when loaded from TOML, so "Dataset"
    / "DATASET" / "dataset" in the config all resolve the same way.
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


def _load_io_settings(raw: dict[str, Any], section_name: str) -> IOSettings:
    section = raw.get(section_name)
    if not section:
        raise RuntimeError(f"Missing required config section: [{section_name}]")
    if "kind" not in section:
        raise RuntimeError(f"Missing required config key: [{section_name}].kind")
    options = _resolve_env_placeholders(section.get("options", {}), f"{section_name}.options")
    return IOSettings(kind=str(section["kind"]).lower(), options=options)


def load_settings(config_path: str | None = None) -> Settings:
    # An empty CICERONE_CONFIG_PATH (e.g. "" from a misconfigured shell/compose
    # file) must fall back to DEFAULT_CONFIG_PATH too, not resolve to the
    # current directory -- hence "or DEFAULT_CONFIG_PATH" rather than relying
    # on os.environ.get's default, which only applies when the var is unset.
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
    )
