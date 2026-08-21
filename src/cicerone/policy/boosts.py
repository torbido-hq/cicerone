"""Item score boosts (see config/features.toml)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd
from rectools import Columns

from cicerone.feature_config import BoostRule
from cicerone.ids import items_id_column
from cicerone.policy.eligibility import warn_missing_column
from cicerone.values import coerce_item_true
from cicerone.values import is_missing as _is_missing

logger = logging.getLogger(__name__)

_warned_boost_without_items = False


def _warn_boost_without_items() -> None:
    global _warned_boost_without_items
    if _warned_boost_without_items:
        return
    _warned_boost_without_items = True
    logger.warning(
        "Boost rules are configured but items data is missing or empty — item boosts will not be applied"
    )


def _boolean_factor(value: object, factor: float) -> float:
    return factor if coerce_item_true(value) else 1.0


def _value_map_factor(value: object, value_factors: dict[str, float]) -> float:
    if _is_missing(value):
        return 1.0
    return float(value_factors.get(str(value), 1.0))


def _numeric_factors(items: pd.DataFrame, column: str, weight: float) -> pd.Series:
    if column not in items.columns or items.empty:
        return pd.Series(1.0, index=items.index)
    series = pd.to_numeric(items[column], errors="coerce")
    lo = series.min(skipna=True)
    hi = series.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        normalized = pd.Series(0.0, index=items.index)
    else:
        normalized = (series - lo) / (hi - lo)
    return 1.0 + weight * normalized.fillna(0.0)


def item_boost_factors(items: pd.DataFrame | None, boosts: Sequence[BoostRule]) -> dict:
    """item_id → product of configured boost factors."""
    if not boosts:
        return {}
    if items is None or items.empty:
        _warn_boost_without_items()
        return {}

    factors = pd.Series(1.0, index=items.index)
    for boost in boosts:
        if boost.item_column not in items.columns:
            warn_missing_column("boost", boost.name, boost.item_column)
            continue
        col = items[boost.item_column]
        if boost.kind == "boolean":
            factors *= col.map(lambda v, f=boost.factor: _boolean_factor(v, f))
        elif boost.kind == "value_map":
            factors *= col.map(lambda v, vf=boost.value_factors: _value_map_factor(v, vf))
        elif boost.kind == "numeric":
            factors *= _numeric_factors(items, boost.item_column, boost.weight)
        else:
            raise ValueError(f"Unknown boost kind {boost.kind!r} in rule {boost.name!r}")

    id_col = items_id_column(items)
    return dict(zip(items[id_col].astype(str), factors.astype(float), strict=True))


def apply_boosts(
    recs: pd.DataFrame,
    items: pd.DataFrame | None,
    boosts: Sequence[BoostRule],
    top_k: int | None = None,
) -> pd.DataFrame:
    """Apply boost multipliers, re-rank; always truncates to ``top_k`` when set."""
    if recs.empty:
        return recs
    if not boosts:
        return _truncate_recs(recs, top_k) if top_k is not None else recs

    factor_by_item = item_boost_factors(items, boosts)
    out = recs.copy()
    if factor_by_item:
        item_ids = out[Columns.Item].astype(str)
        out[Columns.Score] = out[Columns.Score] * item_ids.map(factor_by_item).fillna(1.0)
        out = out.sort_values(
            [Columns.User, Columns.Score, Columns.Item],
            ascending=[True, False, True],
        )
        out[Columns.Rank] = out.groupby(Columns.User).cumcount() + 1
    return _truncate_recs(out, top_k)


def _truncate_recs(recs: pd.DataFrame, top_k: int | None) -> pd.DataFrame:
    if top_k is None:
        return recs.reset_index(drop=True)
    return recs.groupby(Columns.User, as_index=False).head(top_k).reset_index(drop=True)
