"""Declarative eligibility filters and score boosts (see config/features.toml).

Eligibility fail-open / fail-closed matrix (intentional):

- Missing configured ``item_column`` on the items frame → skip that rule
  (fail-open for the rule; other rules still apply).
- Empty allowlist after evaluating rules → skip the cohort (fail-closed;
  no silent full-catalog fallback).
- Missing/empty items frame with user-scoped rules → empty allowlist
  (fail-closed; cannot evaluate item attributes).
- Missing/empty items frame with only item-global rules (e.g. ``item_true``)
  → full catalog (fail-open; cannot filter without item rows).
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Sequence

import pandas as pd
from rectools import Columns

from cicerone.feature_config import BoostRule, EligibilityRule, FeatureConfig
from cicerone.ids import items_id_column
from cicerone.reasons import BOOST_HITS_COLUMN
from cicerone.values import MISSING as _MISSING
from cicerone.values import as_list as _as_list
from cicerone.values import is_missing as _is_missing
from cicerone.values import is_sequence_attr as _is_sequence_attr
from cicerone.values import str_set as _str_set

logger = logging.getLogger(__name__)

_warned_missing_columns: set[tuple[str, str, str]] = set()
_warned_boost_without_items = False

# ``item_true`` string tokens only; avoid ``astype(bool)`` ("false" → True).
_ITEM_TRUE_STRINGS = frozenset({"1", "true", "True", "TRUE", "yes", "Yes", "YES"})
_ITEM_FALSE_STRINGS = frozenset({"0", "false", "False", "FALSE", "no", "No", "NO", ""})


def _warn_missing_column(kind: str, rule_name: str, column: str) -> None:
    key = (kind, rule_name, column)
    if key in _warned_missing_columns:
        return
    _warned_missing_columns.add(key)
    logger.warning(
        "Configured %s rule %r item_column %r not found — skipping rule (fail-open for this rule)",
        kind,
        rule_name,
        column,
    )


def _coerce_item_true(value: object) -> bool:
    """Return True only for explicit truthy tokens; unknowns are False."""
    if _is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    token = str(value).strip()
    if token in _ITEM_TRUE_STRINGS:
        return True
    if token in _ITEM_FALSE_STRINGS:
        return False
    return False


def _item_true_mask(item_values: pd.Series) -> pd.Series:
    """Boolean mask for ``item_true`` without silent string→True coercion."""
    return item_values.map(_coerce_item_true).astype(bool)


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
            mask &= _item_true_mask(item_values)
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
    id_col = items_id_column(items)
    allowed = set(items.loc[mask, id_col].astype(str))
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
    if isinstance(value, bool):
        return factor if value else 1.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return factor if value != 0 else 1.0
    token = str(value).strip()
    if token in _ITEM_TRUE_STRINGS or token.lower() in {"true", "1", "yes"}:
        return factor
    return 1.0


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


def _boost_factor_series(items: pd.DataFrame, boost: BoostRule) -> pd.Series | None:
    if boost.item_column not in items.columns:
        _warn_missing_column("boost", boost.name, boost.item_column)
        return None
    col = items[boost.item_column]
    if boost.kind == "boolean":
        return col.map(lambda v, f=boost.factor: _boolean_factor(v, f)).astype(float)
    if boost.kind == "value_map":
        return col.map(lambda v, vf=boost.value_factors: _value_map_factor(v, vf)).astype(float)
    if boost.kind == "numeric":
        return _numeric_factors(items, boost.item_column, boost.weight).astype(float)
    raise ValueError(f"Unknown boost kind {boost.kind!r} in rule {boost.name!r}")


def item_boost_details(
    items: pd.DataFrame | None, boosts: Sequence[BoostRule]
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
) -> pd.DataFrame:
    """Apply boost multipliers, re-rank; always truncates to ``top_k`` when set."""
    if recs.empty:
        return recs
    if not boosts:
        return _truncate_recs(recs, top_k) if top_k is not None else recs

    factor_by_item, hits_by_item = item_boost_details(items, boosts)
    out = recs.copy()
    item_ids = out[Columns.Item].astype(str)
    out[BOOST_HITS_COLUMN] = [hits_by_item.get(str(item_id), []) for item_id in item_ids]
    if factor_by_item:
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
