from __future__ import annotations

import pandas as pd
import pytest
from rectools import Columns

from cicerone.feature_config import BoostRule, EligibilityRule, FeatureConfig
from cicerone.policy import (
    allowed_items_for_cohort,
    apply_boosts,
    cohort_key,
    eligible_item_mask,
    group_users_by_cohort,
    has_user_scoped_eligibility,
    item_boost_factors,
    resolve_eligibility,
)


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "i1",
                "published": True,
                "in_stock": True,
                "market": "eu",
                "available_countries": ["IT", "FR"],
                "category": "beer",
                "is_paying_producer": True,
                "plan_tier": "premium",
                "margin": 10.0,
            },
            {
                "item_id": "i2",
                "published": True,
                "in_stock": True,
                "market": "us",
                "available_countries": ["US"],
                "category": "wine",
                "is_paying_producer": False,
                "plan_tier": "free",
                "margin": 5.0,
            },
            {
                "item_id": "i3",
                "published": True,
                "in_stock": False,
                "market": "eu",
                "available_countries": ["IT"],
                "category": "beer",
                "is_paying_producer": True,
                "plan_tier": "standard",
                "margin": 20.0,
            },
            {
                "item_id": "i4",
                "published": False,
                "in_stock": True,
                "market": "eu",
                "available_countries": ["DE"],
                "category": "spirits",
                "is_paying_producer": False,
                "plan_tier": "free",
                "margin": 1.0,
            },
        ]
    )


def _users() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "nationality": "IT", "market": "eu", "allowed_categories": ["beer", "wine"]},
            {"user_id": "u2", "nationality": "US", "market": "us", "allowed_categories": ["wine"]},
            {"user_id": "u3", "nationality": None, "market": "eu", "allowed_categories": ["beer"]},
        ]
    )


def _base_config(**overrides) -> FeatureConfig:
    base = dict(
        event_weights={},
        quantity_scaled_events=set(),
        event_caps={},
        user_features=[],
        item_features=[],
        item_availability_filters=["published", "in_stock"],
        eligibility=[],
        boosts=[],
    )
    base.update(overrides)
    return FeatureConfig(**base)


def test_resolve_eligibility_expands_availability_filters():
    config = _base_config(
        eligibility=[
            EligibilityRule(
                name="ships_to_user",
                op="user_in_item_list",
                item_column="available_countries",
                user_column="nationality",
            )
        ]
    )
    rules = resolve_eligibility(config)
    assert [r.name for r in rules] == ["availability:published", "availability:in_stock", "ships_to_user"]
    assert has_user_scoped_eligibility(rules)


def test_eligible_item_mask_item_true():
    items = _items()
    rules = [EligibilityRule(name="pub", op="item_true", item_column="published")]
    mask = eligible_item_mask(None, items, rules)
    assert set(items.loc[mask, "item_id"]) == {"i1", "i2", "i3"}


def test_eligible_item_mask_eq():
    items = _items()
    user = {"user_id": "u1", "market": "eu"}
    rules = [EligibilityRule(name="same_market", op="eq", item_column="market", user_column="market")]
    mask = eligible_item_mask(user, items, rules)
    assert set(items.loc[mask, "item_id"]) == {"i1", "i3", "i4"}


def test_eligible_item_mask_user_in_item_list():
    items = _items()
    user = {"nationality": "IT"}
    rules = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    mask = eligible_item_mask(user, items, rules)
    assert set(items.loc[mask, "item_id"]) == {"i1", "i3"}


def test_eligible_item_mask_item_in_user_list():
    items = _items()
    user = {"allowed_categories": ["beer", "wine"]}
    rules = [
        EligibilityRule(
            name="cats",
            op="item_in_user_list",
            item_column="category",
            user_column="allowed_categories",
        )
    ]
    mask = eligible_item_mask(user, items, rules)
    assert set(items.loc[mask, "item_id"]) == {"i1", "i2", "i3"}


def test_eligible_item_mask_missing_user_exclude_vs_allow():
    items = _items()
    user = {"nationality": None}
    exclude_rule = EligibilityRule(
        name="ships",
        op="user_in_item_list",
        item_column="available_countries",
        user_column="nationality",
        on_missing_user="exclude",
    )
    allow_rule = EligibilityRule(
        name="ships",
        op="user_in_item_list",
        item_column="available_countries",
        user_column="nationality",
        on_missing_user="allow",
    )
    assert not eligible_item_mask(user, items, [exclude_rule]).any()
    assert eligible_item_mask(user, items, [allow_rule]).all()


def test_eligible_item_mask_missing_item_column_fails_open(caplog):
    items = _items()
    rules = [EligibilityRule(name="x", op="item_true", item_column="not_a_column")]
    mask = eligible_item_mask(None, items, rules)
    assert mask.all()
    assert "not_a_column" in caplog.text


def test_missing_column_warnings_are_deduplicated(caplog):
    import cicerone.policy as policy

    policy._warned_missing_columns.clear()
    items = _items()
    rules = [EligibilityRule(name="x", op="item_true", item_column="not_a_column")]
    boosts = [BoostRule(name="y", kind="boolean", item_column="also_missing", factor=2.0)]

    eligible_item_mask(None, items, rules)
    eligible_item_mask(None, items, rules)
    item_boost_factors(items, boosts)
    item_boost_factors(items, boosts)

    assert caplog.text.count("not_a_column") == 1
    assert caplog.text.count("also_missing") == 1


def test_cohort_key_accepts_list_valued_user_attrs():
    user = {"allowed_categories": ["beer", "wine"]}
    rules = [
        EligibilityRule(
            name="cats",
            op="item_in_user_list",
            item_column="category",
            user_column="allowed_categories",
        )
    ]
    assert cohort_key(user, rules) == (("allowed_categories", ("beer", "wine")),)


def test_cohort_key_and_grouping():
    users = _users()
    rules = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    assert cohort_key(users.iloc[0], rules) == (("nationality", "IT"),)
    assert cohort_key(users.iloc[1], rules) == (("nationality", "US"),)
    cohorts = group_users_by_cohort(["u1", "u2", "u1"], users, rules)
    assert [ids for _, ids in cohorts] == [["u1"], ["u2"]]


def test_group_users_by_cohort_keeps_missing_users_under_missing_attr_key():
    users = _users()
    rules = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    cohorts = group_users_by_cohort(["u1", "ghost"], users, rules)
    by_id = {uid: key for key, ids in cohorts for uid in ids}
    assert by_id["u1"] == (("nationality", "IT"),)
    # Absent from users frame → same missing-attr key as a null nationality.
    assert by_id["ghost"] == (("nationality", None),)
    assert set(uid for _, ids in cohorts for uid in ids) == {"u1", "ghost"}


def test_group_users_by_cohort_with_missing_attributes():
    users = _users()
    rules = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    # u3 already has nationality=None in the fixture; share a cohort with a
    # completely unknown user.
    cohorts = group_users_by_cohort(["u1", "u3", "ghost"], users, rules)
    missing_cohort = [ids for key, ids in cohorts if key == (("nationality", None),)][0]
    assert missing_cohort == ["u3", "ghost"]


def test_allowed_items_for_cohort_intersects_availability_and_region():
    config = _base_config(
        eligibility=[
            EligibilityRule(
                name="ships",
                op="user_in_item_list",
                item_column="available_countries",
                user_column="nationality",
            )
        ]
    )
    rules = resolve_eligibility(config)
    allowed = allowed_items_for_cohort(["u1"], _users(), _items(), rules, ["i1", "i2", "i3", "i4"])
    # i1 published+stock and ships to IT; i3 OOS; i2 US-only; i4 unpublished
    assert allowed == ["i1"]


def test_allowed_items_for_cohort_returns_empty_when_nothing_passes(caplog):
    rules = resolve_eligibility(
        _base_config(
            eligibility=[
                EligibilityRule(
                    name="ships",
                    op="user_in_item_list",
                    item_column="available_countries",
                    user_column="nationality",
                )
            ]
        )
    )
    # DE-only unpublished item — no catalog entry survives for an IT user.
    items = pd.DataFrame(
        [
            {
                "item_id": "i4",
                "published": False,
                "in_stock": True,
                "available_countries": ["DE"],
            }
        ]
    )
    allowed = allowed_items_for_cohort(["u1"], _users(), items, rules, ["i4"])
    assert allowed == []
    assert "empty allow-list" in caplog.text


def test_boolean_and_value_map_and_numeric_boosts():
    items = _items()
    boosts = [
        BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0),
        BoostRule(
            name="tier",
            kind="value_map",
            item_column="plan_tier",
            value_factors={"premium": 1.5, "standard": 1.1, "free": 1.0},
        ),
        BoostRule(name="margin", kind="numeric", item_column="margin", weight=1.0),
    ]
    factors = item_boost_factors(items, boosts)
    # i1: 2.0 * 1.5 * (1 + 1*(10-1)/(20-1)) = 3.0 * (1 + 9/19)
    assert factors["i1"] == pytest.approx(3.0 * (1 + 9 / 19))
    # i2: 1.0 * 1.0 * (1 + (5-1)/(20-1))
    assert factors["i2"] == pytest.approx(1.0 * (1 + 4 / 19))


def test_apply_boosts_reorders_and_truncates():
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i2", "rank": 1, "score": 10.0, "source": "personalized"},
            {"user_id": "u1", "item_id": "i1", "rank": 2, "score": 9.0, "source": "personalized"},
        ]
    )
    boosts = [BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0)]
    out = apply_boosts(recs, _items(), boosts, top_k=1)
    assert list(out[Columns.Item]) == ["i1"]
    assert list(out[Columns.Rank]) == [1]
    assert out.iloc[0][Columns.Score] == pytest.approx(18.0)
