"""Eligibility allowlists for incremental write-through."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from cicerone.feature_config import FeatureConfig
from cicerone.ids import items_id_column
from cicerone.policy.eligibility import (
    allowed_items_for_cohort,
    group_users_by_cohort,
    has_user_scoped_eligibility,
    index_users_by_id,
    is_user_scoped,
    resolve_eligibility,
)


def incremental_allowlists(
    user_ids: Sequence[str],
    *,
    feature_config: FeatureConfig | None,
    items: pd.DataFrame | None,
    users: pd.DataFrame | None = None,
) -> dict[str, frozenset[str] | None]:
    """Map user_id → allowlist. ``None`` means do not filter (fail-open)."""
    ids = [str(user_id) for user_id in user_ids]
    open_map: dict[str, frozenset[str] | None] = {user_id: None for user_id in ids}
    if not ids or feature_config is None:
        return open_map
    rules = resolve_eligibility(feature_config)
    if not rules:
        return open_map
    apply_rules = list(rules)
    if has_user_scoped_eligibility(rules) and (users is None or users.empty):
        apply_rules = [rule for rule in rules if not is_user_scoped(rule)]
        if not apply_rules:
            return open_map
    if items is None or items.empty:
        if has_user_scoped_eligibility(apply_rules):
            blocked: frozenset[str] = frozenset()
            return {user_id: blocked for user_id in ids}
        return open_map
    catalog = list(items[items_id_column(items)].astype(str))
    users_by_id = index_users_by_id(users)
    if has_user_scoped_eligibility(apply_rules):
        out: dict[str, frozenset[str] | None] = {}
        for _key, cohort in group_users_by_cohort(ids, users, apply_rules, users_by_id=users_by_id):
            allowed = allowed_items_for_cohort(
                cohort, users, items, apply_rules, catalog, users_by_id=users_by_id
            )
            listed = frozenset(str(item_id) for item_id in allowed)
            for user_id in cohort:
                out[user_id] = listed
        return out
    shared = frozenset(
        str(item_id) for item_id in allowed_items_for_cohort(ids[:1], users, items, apply_rules, catalog)
    )
    return {user_id: shared for user_id in ids}
