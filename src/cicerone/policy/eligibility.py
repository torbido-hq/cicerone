"""Declarative eligibility filters (see config/features.toml).

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

from cicerone.feature_config import EligibilityRule, FeatureConfig
from cicerone.ids import items_id_column
from cicerone.values import MISSING as _MISSING
from cicerone.values import as_list as _as_list
from cicerone.values import is_missing as _is_missing
from cicerone.values import is_sequence_attr as _is_sequence_attr
from cicerone.values import item_true_mask
from cicerone.values import str_set as _str_set

logger = logging.getLogger(__name__)

_warned_missing_columns: set[tuple[str, str, str]] = set()


def warn_missing_column(kind: str, rule_name: str, column: str) -> None:
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
            warn_missing_column("eligibility", rule.name, rule.item_column)
            continue

        item_values = items[rule.item_column]

        if rule.op == "item_true":
            mask &= item_true_mask(item_values)
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
