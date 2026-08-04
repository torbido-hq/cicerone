"""Declarative eligibility filters and score boosts (see config/features.toml)."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
from rectools import Columns

from cicerone.feature_config import BoostRule, EligibilityRule, FeatureConfig

logger = logging.getLogger(__name__)

_MISSING = object()
_warned_missing_columns: set[tuple[str, str, str]] = set()
_warned_boost_without_items = False


def _warn_missing_column(kind: str, rule_name: str, column: str) -> None:
    key = (kind, rule_name, column)
    if key in _warned_missing_columns:
        return
    _warned_missing_columns.add(key)
    logger.warning(
        "Configured %s rule %r item_column %r not found — skipping",
        kind,
        rule_name,
        column,
    )


def _warn_boost_without_items() -> None:
    global _warned_boost_without_items
    if _warned_boost_without_items:
        return
    _warned_boost_without_items = True
    logger.warning(
        "Boost rules are configured but items data is missing or empty — item boosts will not be applied"
    )


def resolve_eligibility(config: FeatureConfig) -> list[EligibilityRule]:
    """Merge item_availability_filters sugar with explicit [[eligibility]] rules."""
    rules: list[EligibilityRule] = []
    already = {(r.op, r.item_column) for r in config.eligibility if r.op == "item_true"}
    for column in config.item_availability_filters:
        if ("item_true", column) not in already:
            rules.append(
                EligibilityRule(
                    name=f"availability:{column}",
                    op="item_true",
                    item_column=column,
                )
            )
    rules.extend(config.eligibility)
    return rules


def is_user_scoped(rule: EligibilityRule) -> bool:
    return rule.op != "item_true"


def has_user_scoped_eligibility(rules: Sequence[EligibilityRule]) -> bool:
    return any(is_user_scoped(r) for r in rules)


def _is_sequence_attr(value: object) -> bool:
    if isinstance(value, (list, tuple, set, pd.Series)):
        return True
    return isinstance(value, np.ndarray) and value.ndim > 0


def _is_missing(value: object) -> bool:
    if value is None or value is _MISSING:
        return True
    if _is_sequence_attr(value):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_list(value: object) -> list:
    if _is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _as_list(value.item())
        values = value.tolist()
    elif isinstance(value, (list, tuple, set, pd.Series)):
        values = list(value) if not isinstance(value, pd.Series) else value.tolist()
    else:
        return [value]
    return [v for v in values if not _is_missing(v)]


def _str_set(values: Iterable) -> set[str]:
    return {str(v) for v in values if not _is_missing(v)}


def _user_attr(user_row: pd.Series | dict | None, column: str | None) -> object:
    if user_row is None or column is None:
        return _MISSING
    if isinstance(user_row, dict):
        if column not in user_row:
            return _MISSING
        return user_row[column]
    if column not in user_row.index:
        return _MISSING
    return user_row[column]


def _user_in_item_list_mask(item_values: pd.Series, user_value: object) -> pd.Series:
    needle = str(user_value)
    exploded = item_values.map(_as_list).explode()
    if exploded.empty:
        return pd.Series(False, index=item_values.index)
    present = exploded[~exploded.map(_is_missing)]
    if present.empty:
        return pd.Series(False, index=item_values.index)
    matches = present.astype(str).eq(needle)
    return matches.groupby(level=0).any().reindex(item_values.index, fill_value=False)


def _item_in_user_list_mask(item_values: pd.Series, user_value: object) -> pd.Series:
    allowed = _str_set(_as_list(user_value))
    non_missing = ~item_values.map(_is_missing)
    return non_missing & item_values.astype(str).isin(allowed)


def eligible_item_mask(
    user_row: pd.Series | dict | None,
    items: pd.DataFrame,
    rules: Sequence[EligibilityRule],
) -> pd.Series:
    """Boolean mask over ``items`` for rows that pass every eligibility rule."""
    mask = pd.Series(True, index=items.index)
    if items.empty or not rules:
        return mask

    for rule in rules:
        if rule.item_column not in items.columns:
            _warn_missing_column("eligibility", rule.name, rule.item_column)
            continue

        item_values = items[rule.item_column]

        if rule.op == "item_true":
            mask &= item_values.fillna(False).astype(bool)
            continue

        user_value = _user_attr(user_row, rule.user_column)
        if _is_missing(user_value):
            if rule.on_missing_user == "allow":
                continue
            mask &= False
            continue

        if rule.op == "eq":
            mask &= item_values.astype(str).eq(str(user_value))
        elif rule.op == "user_in_item_list":
            mask &= _user_in_item_list_mask(item_values, user_value)
        elif rule.op == "item_in_user_list":
            mask &= _item_in_user_list_mask(item_values, user_value)
        else:
            raise ValueError(f"Unknown eligibility op {rule.op!r} in rule {rule.name!r}")

    return mask


def cohort_key(user_row: pd.Series | dict | None, rules: Sequence[EligibilityRule]) -> Hashable:
    """Fingerprint of user attrs used by eligibility (same key → same allowlist)."""
    parts: list[tuple[str, Hashable]] = []
    for rule in rules:
        if not is_user_scoped(rule) or rule.user_column is None:
            continue
        value = _user_attr(user_row, rule.user_column)
        if _is_missing(value):
            parts.append((rule.user_column, None))
        elif _is_sequence_attr(value):
            parts.append((rule.user_column, tuple(sorted(_str_set(_as_list(value))))))
        else:
            parts.append((rule.user_column, str(value)))
    return tuple(parts)


def index_users_by_id(users: pd.DataFrame | None) -> dict[str, pd.Series]:
    """O(1) user_id → row map (first row wins on duplicates)."""
    if users is None or users.empty or "user_id" not in users.columns:
        return {}
    keys = users["user_id"].astype(str)
    keep = ~keys.duplicated(keep="first")
    return {key: users.loc[idx] for idx, key in zip(users.index[keep], keys[keep], strict=True)}


def _user_row_for(
    users: pd.DataFrame | None,
    user_id: str,
    *,
    users_by_id: dict[str, pd.Series] | None = None,
) -> pd.Series | None:
    lookup = users_by_id if users_by_id is not None else index_users_by_id(users)
    return lookup.get(str(user_id))


def allowed_items_for_cohort(
    users_slice: Sequence[str],
    users: pd.DataFrame | None,
    items: pd.DataFrame | None,
    rules: Sequence[EligibilityRule],
    catalog_ids: Iterable,
    *,
    users_by_id: dict[str, pd.Series] | None = None,
) -> list:
    """Recommendable item ids for a cohort (representative = first user in slice).

    Missing/empty ``items``: user-scoped rules → []; item-global-only → full catalog.
    All items filtered out → [] (no silent catalog fallback).
    """
    catalog = list(catalog_ids)
    if not rules:
        return catalog
    if items is None or items.empty:
        if has_user_scoped_eligibility(rules):
            logger.warning(
                "User-scoped eligibility rules are configured but items frame is missing or empty — "
                "returning an empty allowlist (cannot evaluate item attributes)"
            )
            return []
        logger.warning(
            "Item eligibility rules are configured but items frame is missing or empty — "
            "skipping item filters and returning the full catalog"
        )
        return catalog

    lookup = users_by_id if users_by_id is not None else index_users_by_id(users)
    representative = _user_row_for(users, users_slice[0], users_by_id=lookup) if users_slice else None
    mask = eligible_item_mask(representative, items, rules)
    allowed = set(items.loc[mask, "item_id"].astype(str))
    filtered = [i for i in catalog if str(i) in allowed]
    if not filtered and catalog:
        logger.warning(
            "Eligibility rules excluded every item for cohort representative %r — "
            "returning an empty allowlist (no fallback to full catalog)",
            users_slice[0] if users_slice else None,
        )
    return filtered


def group_users_by_cohort(
    target_users: Sequence[str],
    users: pd.DataFrame | None,
    rules: Sequence[EligibilityRule],
    *,
    users_by_id: dict[str, pd.Series] | None = None,
) -> list[tuple[Hashable, list[str]]]:
    """Group target users by eligibility cohort key (missing users kept under missing-attr key)."""
    cohorts: dict[Hashable, list[str]] = {}
    order: list[Hashable] = []
    lookup = users_by_id if users_by_id is not None else index_users_by_id(users)
    for user_id in dict.fromkeys(target_users):
        key = cohort_key(_user_row_for(users, user_id, users_by_id=lookup), rules)
        if key not in cohorts:
            cohorts[key] = []
            order.append(key)
        cohorts[key].append(user_id)
    return [(key, cohorts[key]) for key in order]


def _boolean_factor(value: object, factor: float) -> float:
    if _is_missing(value):
        return 1.0
    return factor if bool(value) else 1.0


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
            _warn_missing_column("boost", boost.name, boost.item_column)
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

    return dict(zip(items["item_id"].astype(str), factors.astype(float), strict=True))


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
