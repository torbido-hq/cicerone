"""Cohort plan for recommend (eligibility groups, overfetch k)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from cicerone.dataset import BuiltDataset
from cicerone.feature_config import DEFAULT_BOOST_OVERFETCH_FACTOR, FeatureConfig
from cicerone.ids import interacting_external_user_ids
from cicerone.model.strategies import STRATEGIES
from cicerone.policy import (
    allowed_items_for_cohort,
    group_users_by_cohort,
    has_user_scoped_eligibility,
    index_users_by_id,
    is_user_scoped,
    resolve_eligibility,
)

logger = logging.getLogger(__name__)


def boost_overfetch_k(
    top_k: int, has_boosts: bool, overfetch_factor: int = DEFAULT_BOOST_OVERFETCH_FACTOR
) -> int:
    if not has_boosts:
        return top_k
    factor = overfetch_factor if overfetch_factor >= 1 else DEFAULT_BOOST_OVERFETCH_FACTOR
    return max(top_k, top_k * factor)


@dataclass(frozen=True)
class _CohortPlan:
    cohorts: list[tuple[object, list[str]]]
    allowed_by_cohort: dict[object, list]
    eligibility: list
    users_frame: pd.DataFrame | None
    known_users: set
    interacting_users: set
    has_any_warm_user: bool
    unique_target_users: list[str]
    all_item_ids: Sequence
    has_boosts: bool
    recommend_k: int


def _resolve_cohort_plan(
    built: BuiltDataset,
    config: FeatureConfig,
    target_users: list[str],
    recommend_models: list[str],
    top_k: int,
) -> _CohortPlan:
    dataset = built.dataset
    all_item_ids = dataset.item_id_map.external_ids
    eligibility = resolve_eligibility(config)
    users_frame = built.users if built.users is not None and not built.users.empty else None
    if has_user_scoped_eligibility(eligibility) and users_frame is None:
        logger.warning(
            "User-scoped eligibility rules are configured but no users frame is available — "
            "applying only item-global rules"
        )
        eligibility = [r for r in eligibility if not is_user_scoped(r)]
    use_cohorts = has_user_scoped_eligibility(eligibility) and users_frame is not None
    has_boosts = bool(config.boosts)
    recommend_k = boost_overfetch_k(top_k, has_boosts, config.boost_overfetch_factor)

    known_users = {str(user_id) for user_id in dataset.user_id_map.external_ids}
    unique_target_users = list(dict.fromkeys(target_users))
    has_any_warm_user = any(str(user_id) in known_users for user_id in unique_target_users)
    needs_interacting_users = any(
        STRATEGIES[name].requires_interactions for name in recommend_models if name in STRATEGIES
    )
    interacting_users = interacting_external_user_ids(built) if needs_interacting_users else set()

    users_by_id = index_users_by_id(users_frame)
    if use_cohorts:
        cohorts = group_users_by_cohort(
            unique_target_users, users_frame, eligibility, users_by_id=users_by_id
        )
        allowed_by_cohort = {
            key: allowed_items_for_cohort(
                cohort_users,
                users_frame,
                built.items,
                eligibility,
                all_item_ids,
                users_by_id=users_by_id,
            )
            for key, cohort_users in cohorts
        }
    else:
        allowed_by_cohort = {
            None: allowed_items_for_cohort(
                unique_target_users,
                users_frame,
                built.items,
                eligibility,
                all_item_ids,
                users_by_id=users_by_id,
            )
        }
        cohorts = [(None, unique_target_users)]

    return _CohortPlan(
        cohorts=cohorts,
        allowed_by_cohort=allowed_by_cohort,
        eligibility=eligibility,
        users_frame=users_frame,
        known_users=known_users,
        interacting_users=interacting_users,
        has_any_warm_user=has_any_warm_user,
        unique_target_users=unique_target_users,
        all_item_ids=all_item_ids,
        has_boosts=has_boosts,
        recommend_k=recommend_k,
    )
