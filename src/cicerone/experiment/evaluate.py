"""Join assignment + events (+ optional exposures) into a sequential experiment report."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cicerone.config.settings import ExperimentSettings
from cicerone.experiment.assignment import assign_variant
from cicerone.experiment.guardrails import GuardrailReport, evaluate_guardrails
from cicerone.experiment.recipes import ResolvedRecipe
from cicerone.experiment.stats import ComparisonResult, compare_variants, pick_control_name, variant_metric
from cicerone.io.recommendation_schema import USER_COLUMN, VARIANT_COLUMN, filter_variant_rows

PRIMARY_METRIC_WEIGHTED = "weighted"


@dataclass(frozen=True)
class ExperimentReport:
    experiment_id: str
    primary_metric: str
    comparisons: tuple[ComparisonResult, ...]
    guardrails: tuple[GuardrailReport, ...]
    promoted_variant: str | None
    n_assigned: int
    exposure_conditional: bool
    can_promote: bool
    promote_blocked_by: tuple[str, ...] = field(default_factory=tuple)

    @property
    def winner(self) -> str | None:
        if not self.can_promote:
            return None
        winners = {item.winner for item in self.comparisons if item.decided}
        if len(winners) == 1:
            return next(iter(winners))
        return None


def user_outcome(
    events: pd.DataFrame,
    *,
    event_weights: dict[str, float],
    primary_metric: str,
) -> dict[str, float]:
    """Map user_id → outcome (weighted sum, or count of ``primary_metric`` events)."""
    if events.empty or USER_COLUMN not in events.columns:
        return {}
    frame = events.copy()
    frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
    if "event_type" not in frame.columns:
        return {}
    if primary_metric != PRIMARY_METRIC_WEIGHTED:
        matched = frame[frame["event_type"].astype(str) == primary_metric]
        counts = matched.groupby(USER_COLUMN).size()
        return {str(user_id): float(count) for user_id, count in counts.items()}
    weights = frame["event_type"].astype(str).map(lambda name: float(event_weights.get(name, 0.0)))
    if "quantity" in frame.columns:
        quantity = pd.to_numeric(frame["quantity"], errors="coerce").fillna(1.0)
        weights = weights * quantity
    totals = pd.DataFrame({USER_COLUMN: frame[USER_COLUMN], "_w": weights}).groupby(USER_COLUMN)["_w"].sum()
    return {str(user_id): float(total) for user_id, total in totals.items()}


def evaluate_experiment(
    *,
    experiment: ExperimentSettings,
    recipes: tuple[ResolvedRecipe, ...],
    events: pd.DataFrame,
    event_weights: dict[str, float],
    recommendations: pd.DataFrame | None = None,
    exposures: list[dict[str, Any]] | None = None,
    promoted_variant: str | None = None,
    catalog_size: int | None = None,
) -> ExperimentReport:
    variants = [(recipe.name, recipe.traffic) for recipe in recipes]
    names = [recipe.name for recipe in recipes]
    assigned: dict[str, str] = {}
    exposure_conditional = bool(exposures)
    if exposures:
        for row in exposures:
            user_id = str(row.get("user_id") or "")
            variant = str(row.get("variant") or "")
            if user_id and variant in names:
                assigned[user_id] = variant
    else:
        if not events.empty and USER_COLUMN in events.columns:
            for user_id in events[USER_COLUMN].astype(str).unique():
                assigned[user_id] = assign_variant(
                    experiment.id,
                    str(user_id),
                    variants,
                )
    outcomes = user_outcome(events, event_weights=event_weights, primary_metric=experiment.primary_metric)
    values_by_variant: dict[str, list[float]] = defaultdict(list)
    for user_id, variant in assigned.items():
        values_by_variant[variant].append(float(outcomes.get(user_id, 0.0)))
    for name in names:
        values_by_variant.setdefault(name, [])
    control_name = pick_control_name(names)
    control = variant_metric(control_name, values_by_variant[control_name])
    comparisons: list[ComparisonResult] = []
    for name in names:
        if name == control_name:
            continue
        comparisons.append(
            compare_variants(
                control,
                variant_metric(name, values_by_variant[name]),
                alpha=experiment.alpha,
            )
        )
    guardrails: list[GuardrailReport] = []
    if recommendations is not None and VARIANT_COLUMN in recommendations.columns:
        for name in names:
            slice_rows = filter_variant_rows(recommendations, name)
            guardrails.append(evaluate_guardrails(slice_rows, variant=name, catalog_size=catalog_size))
    blocked: list[str] = []
    if any(not item.ok for item in guardrails):
        blocked.append("guardrails")
    decided = comparisons and all(item.decided for item in comparisons)
    winners = {item.winner for item in comparisons if item.decided}
    if not decided:
        blocked.append("undecided")
    elif len(winners) != 1:
        blocked.append("split_winners")
    return ExperimentReport(
        experiment_id=experiment.id,
        primary_metric=experiment.primary_metric,
        comparisons=tuple(comparisons),
        guardrails=tuple(guardrails),
        promoted_variant=promoted_variant,
        n_assigned=len(assigned),
        exposure_conditional=exposure_conditional,
        can_promote=not blocked,
        promote_blocked_by=tuple(blocked),
    )


def exposure_row(
    *,
    user_id: str,
    experiment_id: str,
    variant: str,
    generated_at: str | None,
    exposed_at: datetime | None = None,
) -> dict[str, Any]:
    when = exposed_at or datetime.now(UTC)
    return {
        "user_id": user_id,
        "experiment_id": experiment_id,
        "variant": variant,
        "generated_at": generated_at,
        "exposed_at": when.isoformat(),
    }
