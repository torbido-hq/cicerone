"""Config validation helpers, number fields and the epoch-metrics resolver."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from cicerone.config.constants import (
    DEFAULT_EPOCH_METRICS_EVERY,
    DEFAULT_EPOCH_METRICS_MAX_USERS,
    DEFAULT_EPOCH_METRICS_PLATEAU_EPS,
    DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW,
    DEFAULT_EPOCH_METRICS_REGRESSION_DROP,
    ConfigError,
)
from cicerone.config.settings import EpochMetricsSettings

_T = TypeVar("_T")

_ENV_PLACEHOLDER = re.compile(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_env_placeholders(value: Any, path: str = "") -> Any:
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
            key: resolve_env_placeholders(item, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_env_placeholders(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def load_kind_section(
    raw: Mapping[str, Any], section: str, default_kind: str, kinds: Sequence[str]
) -> tuple[bool, str, dict[str, Any]]:
    """``enabled``, ``kind`` and resolved ``options`` of an ``[events]``/``[publish]``-style table."""
    enabled = bool(raw.get("enabled", False))
    kind = str(raw.get("kind", default_kind)).lower()
    if enabled and kind not in kinds:
        raise ConfigError(f"{section}.kind must be one of {list(kinds)}, got {kind!r}")
    options = resolve_env_placeholders(raw.get("options", {}), f"{section}.options")
    if not isinstance(options, dict):
        raise ConfigError(f"{section}.options must be a table")
    return enabled, kind, options


def int_field(raw: Mapping[str, Any], name: str, default: int, require: Callable[..., int]) -> int:
    """Coerce ``raw[key]`` (``key`` = last dotted segment of ``name``) and check it with ``require``."""
    return require(int(raw.get(name.rpartition(".")[2], default)), name=name)


def float_field(raw: Mapping[str, Any], name: str, default: float, require: Callable[..., float]) -> float:
    return require(float(raw.get(name.rpartition(".")[2], default)), name=name)


def resolve_epoch_metrics(job: Mapping[str, Any]) -> EpochMetricsSettings | None:
    """Epoch-metric settings, or ``None`` when logging is off."""
    if not bool(job.get("log_epoch_metrics", False)):
        return None
    return EpochMetricsSettings(
        every=int_field(job, "job.epoch_metrics_every", DEFAULT_EPOCH_METRICS_EVERY, require_positive_int),
        max_users=int_field(
            job, "job.epoch_metrics_max_users", DEFAULT_EPOCH_METRICS_MAX_USERS, require_positive_int
        ),
        regression_drop=float_field(
            job,
            "job.epoch_metrics_regression_drop",
            DEFAULT_EPOCH_METRICS_REGRESSION_DROP,
            require_unit_interval,
        ),
        plateau_eps=float_field(
            job, "job.epoch_metrics_plateau_eps", DEFAULT_EPOCH_METRICS_PLATEAU_EPS, require_unit_interval
        ),
        plateau_window=int_field(
            job,
            "job.epoch_metrics_plateau_window",
            DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW,
            require_positive_int,
        ),
    )


def require_finite(value: float, *, name: str) -> float:
    """Reject NaN and infinity, which TOML accepts and every range comparison lets through."""
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number, got {value}")
    return value


def validate_model_weights(weights: dict[str, float] | None, *, context: str = "model_weights") -> None:
    if weights is None:
        return
    for name, weight in weights.items():
        require_finite(weight, name=f"{context}.{name}")
    negative_weights = {name: weight for name, weight in weights.items() if weight < 0}
    if negative_weights:
        raise ConfigError(f"{context} value(s) must be non-negative, got {negative_weights}")


def validate_rrf_k(rrf_k: float | None, *, context: str = "rrf_k") -> None:
    if rrf_k is not None:
        require_finite(rrf_k, name=context)
    if rrf_k is not None and rrf_k <= 0:
        raise ConfigError(f"{context} must be positive, got {rrf_k}")


def require_positive_int(value: int, *, name: str) -> int:
    if value < 1:
        raise ConfigError(f"{name} must be >= 1, got {value}")
    return value


def require_non_negative_int(value: int, *, name: str) -> int:
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}")
    return value


def require_positive_float(value: float, *, name: str) -> float:
    require_finite(value, name=name)
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value


def require_open_unit_interval(value: float, *, name: str) -> float:
    """Require a relative fraction in ``(0, 1)``."""
    require_finite(value, name=name)
    if value <= 0 or value >= 1:
        raise ConfigError(f"{name} must be in (0, 1), got {value}")
    return value


def require_unit_interval(value: float, *, name: str) -> float:
    """Require a relative fraction in (0, 1]."""
    require_finite(value, name=name)
    if value <= 0 or value > 1:
        raise ConfigError(f"{name} must be in (0, 1], got {value}")
    return value


def unique_policy_names(names: Sequence[object], *, label: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            raise ConfigError(f"{label} rule name must be non-empty")
        if name in seen:
            raise ConfigError(f"{label} duplicate rule name {name!r}")
        seen.add(name)
        out.append(name)
    return tuple(out)


def parse_policy_rules(
    items: Sequence[Mapping[str, Any]],
    parse: Callable[[Sequence[Mapping[str, Any]]], Sequence[_T]],
    *,
    label: str,
) -> tuple[_T, ...]:
    try:
        rules = tuple(parse([dict(item) for item in items]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{label}: {exc}") from exc
    unique_policy_names([rule.name for rule in rules], label=label)  # type: ignore[attr-defined]
    return rules
