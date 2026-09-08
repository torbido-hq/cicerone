"""Approximate mixture interval for a two-sample mean difference."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cicerone.experiment.recipes import CONTROL_NAME

_LOG_LOG_FLOOR = 3.0
_VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class VariantMetric:
    name: str
    n_users: int
    conversions: int
    total: float
    mean: float
    variance: float


@dataclass(frozen=True)
class ComparisonResult:
    control: VariantMetric
    treatment: VariantMetric
    difference: float
    ci_low: float
    ci_high: float
    decided: bool
    winner: str | None
    alpha: float


def variant_metric(name: str, values: list[float]) -> VariantMetric:
    n = len(values)
    total = float(sum(values))
    conversions = sum(1 for value in values if value > 0)
    mean = total / n if n else 0.0
    variance = 0.0 if n < 2 else sum((value - mean) ** 2 for value in values) / (n - 1)
    return VariantMetric(
        name=name,
        n_users=n,
        conversions=conversions,
        total=total,
        mean=mean,
        variance=variance,
    )


def mixing_radius(n: int, variance: float, *, alpha: float) -> float:
    """Robbins–Siegmund style mixture bound for a sample mean."""
    if n < 2 or alpha <= 0 or alpha >= 1:
        return math.inf
    t = max(float(n), _LOG_LOG_FLOOR)
    inner = math.log(math.log(t)) + math.log(2.0 / alpha)
    return math.sqrt(2.0 * max(variance, _VARIANCE_FLOOR) * max(inner, 1e-9) / n)


def compare_variants(
    control: VariantMetric,
    treatment: VariantMetric,
    *,
    alpha: float,
) -> ComparisonResult:
    difference = treatment.mean - control.mean
    radius = math.hypot(
        mixing_radius(control.n_users, control.variance, alpha=alpha),
        mixing_radius(treatment.n_users, treatment.variance, alpha=alpha),
    )
    ci_low = difference - radius
    ci_high = difference + radius
    decided = math.isfinite(radius) and (ci_high < 0 or ci_low > 0)
    winner: str | None = None
    if decided:
        winner = treatment.name if difference > 0 else control.name
    return ComparisonResult(
        control=control,
        treatment=treatment,
        difference=difference,
        ci_low=ci_low,
        ci_high=ci_high,
        decided=decided,
        winner=winner,
        alpha=alpha,
    )


def pick_control_name(names: list[str]) -> str:
    if CONTROL_NAME in names:
        return CONTROL_NAME
    return names[0]
