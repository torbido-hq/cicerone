"""Item score boosts (see config/features.toml)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd
from rectools import Columns

from cicerone.feature_config import BoostRule
from cicerone.ids import items_id_column
from cicerone.policy.eligibility import warn_missing_column
from cicerone.reasons import BOOST_HITS_COLUMN
from cicerone.values import is_missing as _is_missing
from cicerone.values import item_true_mask

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


def _boost_factor_series(items: pd.DataFrame, boost: BoostRule) -> pd.Series | None:
    if boost.item_column not in items.columns:
        warn_missing_column("boost", boost.name, boost.item_column)
        return None
    col = items[boost.item_column]
    if boost.kind == "boolean":
        return item_true_mask(col).astype(float) * (boost.factor - 1.0) + 1.0
    if boost.kind == "value_map":
        missing = col.map(_is_missing)
        mapped = col.astype(str).map(boost.value_factors)
        return mapped.where(~missing, 1.0).fillna(1.0).astype(float)
    if boost.kind == "numeric":
        return _numeric_factors(items, boost.item_column, boost.weight).astype(float)
    raise ValueError(f"Unknown boost kind {boost.kind!r} in rule {boost.name!r}")


def item_boost_details(
    items: pd.DataFrame | None,
    boosts: Sequence[BoostRule],
    *,
    record_hits: bool = True,
) -> tuple[dict[str, float], dict[str, list[dict[str, object]]]]:
    """item_id → (product of factors, hits where a rule factor was not 1.0)."""
    if not boosts:
        return {}, {}
    if items is None or items.empty:
        _warn_boost_without_items()
        return {}, {}

    product = pd.Series(1.0, index=items.index)
    rule_series: list[tuple[str, pd.Series]] = []
    for boost in boosts:
        series = _boost_factor_series(items, boost)
        if series is None:
            continue
        product *= series
        rule_series.append((boost.name, series))

    id_col = items_id_column(items)
    ids = items[id_col].astype(str)
    factors = dict(zip(ids, product.astype(float), strict=True))
    if not record_hits:
        return factors, {}
    hits: dict[str, list[dict[str, object]]] = {}
    for item_id, idx in zip(ids, items.index, strict=True):
        item_hits = []
        for name, series in rule_series:
            factor = float(series.loc[idx])
            if factor != 1.0:
                item_hits.append({"name": name, "factor": factor})
        if item_hits:
            hits[str(item_id)] = item_hits
    return factors, hits


def item_boost_factors(items: pd.DataFrame | None, boosts: Sequence[BoostRule]) -> dict:
    """item_id → product of configured boost factors."""
    factors, _hits = item_boost_details(items, boosts)
    return factors


def apply_boosts(
    recs: pd.DataFrame,
    items: pd.DataFrame | None,
    boosts: Sequence[BoostRule],
    top_k: int | None = None,
    *,
    record_hits: bool = True,
) -> pd.DataFrame:
    """Apply boost multipliers, re-rank; always truncates to ``top_k`` when set."""
    if recs.empty:
        return recs
    if not boosts:
        return _truncate_recs(recs, top_k) if top_k is not None else recs

    factor_by_item, hits_by_item = item_boost_details(items, boosts, record_hits=record_hits)
    out = recs.copy()
    item_ids = out[Columns.Item].astype(str)
    if record_hits:
        out[BOOST_HITS_COLUMN] = [hits_by_item.get(str(item_id), []) for item_id in item_ids]
    if factor_by_item:
        out[Columns.Score] = out[Columns.Score] * item_ids.map(factor_by_item).fillna(1.0)
        out = out.sort_values(
            [Columns.User, Columns.Score, Columns.Item],
            ascending=[True, False, True],
            kind="mergesort",
        )
        out[Columns.Rank] = out.groupby(Columns.User).cumcount() + 1
    return _truncate_recs(out, top_k)


def _truncate_recs(recs: pd.DataFrame, top_k: int | None) -> pd.DataFrame:
    if top_k is None:
        return recs.reset_index(drop=True)
    return recs.groupby(Columns.User, as_index=False).head(top_k).reset_index(drop=True)
