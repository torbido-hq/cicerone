"""Join assignment + events (+ optional exposures) into a sequential experiment report."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cicerone.config.constants import PRIMARY_METRIC_WEIGHTED
from cicerone.config.settings import ExperimentSettings
from cicerone.experiment.assignment import assign_variant
from cicerone.experiment.guardrails import GuardrailReport, evaluate_guardrails
from cicerone.experiment.recipes import ResolvedRecipe
from cicerone.experiment.stats import ComparisonResult, compare_variants, pick_control_name, variant_metric
from cicerone.io.recommendation_schema import USER_COLUMN, VARIANT_COLUMN, filter_variant_rows


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
        return _unique_best_mean(self.comparisons)


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
    weights = frame["event_type"].astype(str).map(event_weights).fillna(0.0).astype(float)
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
    promoted_at: str | None = None,
    catalog_size: int | None = None,
) -> ExperimentReport:
    variants = [(recipe.name, recipe.traffic) for recipe in recipes]
    names = [recipe.name for recipe in recipes]
    assigned: dict[str, str] = {}
    exposure_conditional = exposures is not None
    starts: dict[str, pd.Timestamp] = {}
    if exposures is not None:
        assigned, starts = _first_exposures(exposures, experiment_id=experiment.id, names=names)
    elif not events.empty and USER_COLUMN in events.columns:
        for user_id in events[USER_COLUMN].astype(str).unique():
            assigned[str(user_id)] = assign_variant(
                experiment.id,
                str(user_id),
                variants,
            )
    until, invalid_until = _parse_cutoff(promoted_at)
    if invalid_until:
        metric_events = events.iloc[0:0]
    else:
        metric_events = _restrict_events(events, starts=starts, until=until)
    outcomes = user_outcome(
        metric_events, event_weights=event_weights, primary_metric=experiment.primary_metric
    )
    values_by_variant: dict[str, list[float]] = defaultdict(list)
    for user_id, variant in assigned.items():
        values_by_variant[variant].append(float(outcomes.get(user_id, 0.0)))
    for name in names:
        values_by_variant.setdefault(name, [])
    control_name = pick_control_name(names)
    control = variant_metric(control_name, values_by_variant[control_name])
    comparisons: list[ComparisonResult] = []
    arms = max(1, len(names) - 1)
    alpha = experiment.alpha / arms
    for name in names:
        if name == control_name:
            continue
        comparisons.append(
            compare_variants(
                control,
                variant_metric(name, values_by_variant[name]),
                alpha=alpha,
            )
        )
    guardrails: list[GuardrailReport] = []
    blocked: list[str] = []
    if recommendations is None or VARIANT_COLUMN not in recommendations.columns:
        blocked.append("guardrails")
    else:
        for name in names:
            slice_rows = filter_variant_rows(recommendations, name)
            guardrails.append(evaluate_guardrails(slice_rows, variant=name, catalog_size=catalog_size))
        if any(not item.ok for item in guardrails):
            blocked.append("guardrails")
    decided = bool(comparisons) and all(item.decided for item in comparisons)
    if promoted_variant:
        blocked.append("promoted")
    if invalid_until:
        blocked.append("promoted_at")
    if not decided:
        blocked.append("undecided")
    elif _unique_best_mean(comparisons) is None:
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


def _parse_cutoff(value: object) -> tuple[pd.Timestamp | None, bool]:
    if value is None or value == "":
        return None, False
    stamp = _as_timestamp(value)
    return stamp, stamp is None


def _as_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return stamp


def _first_exposures(
    exposures: list[dict[str, Any]],
    *,
    experiment_id: str,
    names: list[str],
) -> tuple[dict[str, str], dict[str, pd.Timestamp]]:
    assigned: dict[str, str] = {}
    starts: dict[str, pd.Timestamp] = {}
    for row in exposures:
        if str(row.get("experiment_id") or "") != experiment_id:
            continue
        user_id = str(row.get("user_id") or "")
        variant = str(row.get("variant") or "")
        if not user_id or variant not in names:
            continue
        when = _as_timestamp(row.get("exposed_at"))
        previous = starts.get(user_id)
        if user_id not in assigned:
            assigned[user_id] = variant
            if when is not None:
                starts[user_id] = when
        elif when is not None and (previous is None or when < previous):
            assigned[user_id] = variant
            starts[user_id] = when
    return assigned, starts


def _restrict_events(
    events: pd.DataFrame,
    *,
    starts: dict[str, pd.Timestamp],
    until: pd.Timestamp | None,
) -> pd.DataFrame:
    window = until is not None or bool(starts)
    if events.empty or USER_COLUMN not in events.columns:
        return events
    if "occurred_at" not in events.columns:
        return events.iloc[0:0] if window else events
    frame = events.copy()
    frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
    if not window:
        return frame
    occurred = pd.to_datetime(frame["occurred_at"], utc=True, errors="coerce")
    keep = occurred.notna()
    if until is not None:
        keep = keep & (occurred < until)
    if starts:
        start_at = frame[USER_COLUMN].map(starts)
        keep = keep & start_at.notna() & (occurred >= start_at)
    return frame.loc[keep]


def _unique_best_mean(comparisons: tuple[ComparisonResult, ...] | list[ComparisonResult]) -> str | None:
    means: dict[str, float] = {}
    for item in comparisons:
        means[item.control.name] = item.control.mean
        means[item.treatment.name] = item.treatment.mean
    if not means:
        return None
    best = max(means.values())
    names = [name for name, mean in means.items() if mean == best]
    return names[0] if len(names) == 1 else None


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
