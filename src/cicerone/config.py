"""Configuration for the Cicerone recommender job.

Everything is loaded from a single TOML file (default:
/app/config/cicerone.toml, override with CICERONE_CONFIG_PATH). Secrets are
never stored in the TOML file itself: reference them with ${ENV_VAR_NAME}
placeholders, resolved from the process environment at load time (see
.env.example). This keeps the structural configuration (which backend, which
bucket/table, scheduling, tuning...) in version control while credentials
stay in environment variables / secret stores.

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

# Matches every "${VAR_NAME}" occurrence within a string, not just a string
# that's *entirely* one placeholder — so both `bucket = "${BUCKET}"` and
# `prefix = "datasets/${ENV}/latest"` are resolved the same way.
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_placeholders(value: Any) -> Any:
    """Recursively replaces "${VAR_NAME}" occurrences with the matching
    environment variable, so secrets never have to be written into the
    (version-controlled) TOML config file itself. Supports partial
    interpolation (e.g. "datasets/${ENV}/latest"), not just a string that is
    exactly one placeholder."""
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise RuntimeError(f"Config references ${{{name}}} but that environment variable is not set")
            return os.environ[name]

        return _ENV_PLACEHOLDER.sub(_replace, value)
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    return value


@dataclass(frozen=True)
class IOSettings:
    """Generic I/O configuration: a backend "kind" plus its own options.

    Deliberately untyped (``options`` is a plain dict) so new input/output
    backends can be added under cicerone.io without ever touching this
    module — see cicerone.io.factory.build_input_source/build_output_sink.
    """

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


def _load_io_settings(raw: dict[str, Any], section_name: str) -> IOSettings:
    section = raw.get(section_name)
    if not section or "kind" not in section:
        raise RuntimeError(f"Missing required config section: [{section_name}] with a 'kind' key")
    options = _resolve_env_placeholders(section.get("options", {}))
    return IOSettings(kind=section["kind"], options=options)


def load_settings(config_path: str | None = None) -> Settings:
    path = Path(config_path or os.environ.get("CICERONE_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    job = raw.get("job", {})
    return Settings(
        input=_load_io_settings(raw, "input"),
        output=_load_io_settings(raw, "output"),
        feature_config_path=job.get("feature_config_path", "/app/config/features.toml"),
        top_k=int(job.get("top_k", 10)),
        half_life_days=float(job.get("half_life_days", 90)),
        cron_schedule=job.get("cron_schedule", "0 3 * * *"),
    )
