"""Config validation helpers and epoch/worker resolvers."""

from __future__ import annotations

from typing import Any

from cicerone.config.constants import (
    DEFAULT_EPOCH_METRICS_EVERY,
    DEFAULT_EPOCH_METRICS_MAX_USERS,
    DEFAULT_EPOCH_METRICS_PLATEAU_EPS,
    DEFAULT_EPOCH_METRICS_PLATEAU_WINDOW,
    DEFAULT_EPOCH_METRICS_REGRESSION_DROP,
    DEFAULT_MAX_WORKERS,
    ConfigError,
)
from cicerone.config.settings import EpochMetricsSettings


def resolve_max_workers(raw: Any | None = None) -> int:
    """Process-pool size; omit/None → sequential (1)."""
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
    """Epoch-metric settings, or ``None`` when logging is off."""
    if not log_epoch_metrics:
        return None
    return EpochMetricsSettings(
        every=require_positive_int(
            DEFAULT_EPOCH_METRICS_EVERY if every is None else int(every),
            name="job.epoch_metrics_every",
        ),
        max_users=require_positive_int(
            DEFAULT_EPOCH_METRICS_MAX_USERS if max_users is None else int(max_users),
            name="job.epoch_metrics_max_users",
        ),
        regression_drop=require_unit_interval(
            DEFAULT_EPOCH_METRICS_REGRESSION_DROP if regression_drop is None else float(regression_drop),
            name="job.epoch_metrics_regression_drop",
        ),
        plateau_eps=require_unit_interval(
            DEFAULT_EPOCH_METRICS_PLATEAU_EPS if plateau_eps is None else float(plateau_eps),
            name="job.epoch_metrics_plateau_eps",
        ),
        plateau_window=require_positive_int(
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


def require_positive_int(value: int, *, name: str) -> int:
    if value < 1:
        raise ConfigError(f"{name} must be >= 1, got {value}")
    return value


def require_non_negative_int(value: int, *, name: str) -> int:
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}")
    return value


def require_positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value


def require_unit_interval(value: float, *, name: str) -> float:
    """Require a relative fraction in (0, 1]."""
    if value <= 0 or value > 1:
        raise ConfigError(f"{name} must be in (0, 1], got {value}")
    return value
