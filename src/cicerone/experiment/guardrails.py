"""Fail-closed catalog health checks on a variant's recommendation rows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cicerone.blending import LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io.recommendation_schema import ITEM_COLUMN, SOURCE_COLUMN, USER_COLUMN

DEFAULT_MAX_FALLBACK_RATE = 0.5
DEFAULT_MAX_TOP_ITEM_SHARE = 0.4
DEFAULT_MIN_COVERAGE = 5

_FALLBACK_SOURCES = frozenset({POPULAR_SOURCE, LATEST_SOURCE, "incremental"})


@dataclass(frozen=True)
class GuardrailReport:
    variant: str
    fallback_rate: float
    top_item_share: float
    distinct_items: int
    ok: bool
    failures: tuple[str, ...]


def _is_fallback_source(source: str) -> bool:
    if source in _FALLBACK_SOURCES:
        return True
    return all(part in _FALLBACK_SOURCES for part in source.split("+") if part)


def evaluate_guardrails(
    recs: pd.DataFrame,
    *,
    variant: str,
    catalog_size: int | None = None,
    max_fallback_rate: float = DEFAULT_MAX_FALLBACK_RATE,
    max_top_item_share: float = DEFAULT_MAX_TOP_ITEM_SHARE,
    min_coverage: int = DEFAULT_MIN_COVERAGE,
) -> GuardrailReport:
    if recs.empty:
        return GuardrailReport(
            variant=variant,
            fallback_rate=1.0,
            top_item_share=1.0,
            distinct_items=0,
            ok=False,
            failures=("empty_recommendations",),
        )
    n_rows = len(recs)
    if SOURCE_COLUMN in recs.columns:
        fallback_rate = float(recs[SOURCE_COLUMN].astype(str).map(_is_fallback_source).mean())
    else:
        fallback_rate = 0.0
    item_counts = recs[ITEM_COLUMN].astype(str).value_counts() if ITEM_COLUMN in recs.columns else None
    if item_counts is not None and not item_counts.empty:
        top_item_share = float(item_counts.iloc[0] / n_rows)
    else:
        top_item_share = 1.0
    distinct_items = int(item_counts.size) if item_counts is not None else 0
    coverage_floor = min_coverage
    if catalog_size is not None and catalog_size > 0:
        coverage_floor = min(min_coverage, max(1, catalog_size // 20))
    failures: list[str] = []
    if fallback_rate > max_fallback_rate:
        failures.append("fallback_rate")
    if top_item_share > max_top_item_share:
        failures.append("top_item_share")
    if distinct_items < coverage_floor:
        failures.append("coverage")
    return GuardrailReport(
        variant=variant,
        fallback_rate=fallback_rate,
        top_item_share=top_item_share,
        distinct_items=distinct_items,
        ok=not failures,
        failures=tuple(failures),
    )


def users_in_frame(recs: pd.DataFrame) -> set[str]:
    if recs.empty or USER_COLUMN not in recs.columns:
        return set()
    return set(recs[USER_COLUMN].astype(str))
