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
    index_users_by_id,
    item_boost_factors,
    resolve_eligibility,
)
from cicerone.reasons import BOOST_HITS_COLUMN


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


def test_eligible_item_mask_item_true_string_false_is_ineligible():
    items = pd.DataFrame(
        [
            {"item_id": "a", "published": "true"},
            {"item_id": "b", "published": "false"},
            {"item_id": "c", "published": "0"},
            {"item_id": "d", "published": "1"},
            {"item_id": "e", "published": ""},
        ]
    )
    rules = [EligibilityRule(name="pub", op="item_true", item_column="published")]
    mask = eligible_item_mask(None, items, rules)
    assert set(items.loc[mask, "item_id"]) == {"a", "d"}


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


def test_eligible_item_mask_list_ops_normalize_types():
    """Cohort fingerprint and mask both stringify, so int/str list elements agree."""
    items = pd.DataFrame(
        [
            {"item_id": "i1", "region_ids": [1, 2], "category_id": 10},
            {"item_id": "i2", "region_ids": ["3"], "category_id": 20},
        ]
    )
    # user_in_item_list: scalar user attr vs list item attr
    ships = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="region_ids",
            user_column="region",
        )
    ]
    assert set(items.loc[eligible_item_mask({"region": "1"}, items, ships), "item_id"]) == {"i1"}
    assert set(items.loc[eligible_item_mask({"region": 1}, items, ships), "item_id"]) == {"i1"}

    # item_in_user_list: list user attr vs scalar item attr
    cats = [
        EligibilityRule(
            name="cats",
            op="item_in_user_list",
            item_column="category_id",
            user_column="allowed",
        )
    ]
    assert set(items.loc[eligible_item_mask({"allowed": [10, 99]}, items, cats), "item_id"]) == {"i1"}
    assert set(items.loc[eligible_item_mask({"allowed": ["10", "99"]}, items, cats), "item_id"]) == {"i1"}

    # Same stringified list → same cohort key
    assert cohort_key({"allowed": [10, 99]}, cats) == cohort_key({"allowed": ["10", "99"]}, cats)


def test_eligible_item_mask_and_cohort_key_accept_numpy_list_cells():
    """Parquet often surfaces list columns as numpy ndarrays."""
    import numpy as np

    items = pd.DataFrame(
        [
            {"item_id": "i1", "available_countries": np.array(["IT", "FR"], dtype=object)},
            {"item_id": "i2", "available_countries": np.array(["US"], dtype=object)},
        ]
    )
    ships = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    assert set(items.loc[eligible_item_mask({"nationality": "IT"}, items, ships), "item_id"]) == {"i1"}

    cats = [
        EligibilityRule(
            name="cats",
            op="item_in_user_list",
            item_column="category",
            user_column="allowed_categories",
        )
    ]
    items_cats = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i2", "category": "wine"},
        ]
    )
    user = {"allowed_categories": np.array(["beer", "wine"], dtype=object)}
    assert set(items_cats.loc[eligible_item_mask(user, items_cats, cats), "item_id"]) == {"i1", "i2"}
    assert cohort_key(user, cats) == cohort_key({"allowed_categories": ["beer", "wine"]}, cats)
    assert cohort_key(user, cats) == (("allowed_categories", ("beer", "wine")),)


def test_index_users_by_id_first_row_wins_across_types():
    users = pd.DataFrame(
        [
            {"user_id": 1, "nationality": "IT"},
            {"user_id": "1", "nationality": "US"},
            {"user_id": 2, "nationality": "DE"},
        ]
    )
    indexed = index_users_by_id(users)
    assert set(indexed) == {"1", "2"}
    assert indexed["1"]["nationality"] == "IT"
    assert indexed["2"]["nationality"] == "DE"


def test_apply_boosts_tie_break_is_deterministic():
    items = pd.DataFrame(
        [
            {"item_id": "i_b", "is_paying_producer": True},
            {"item_id": "i_a", "is_paying_producer": True},
        ]
    )
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i_b", "rank": 1, "score": 1.0, "source": "popular_fallback"},
            {"user_id": "u1", "item_id": "i_a", "rank": 2, "score": 1.0, "source": "popular_fallback"},
        ]
    )
    boosts = [BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0)]
    out = apply_boosts(recs, items, boosts, top_k=2)
    # Equal boosted scores → stable sort by item id ascending
    assert list(out[Columns.Item]) == ["i_a", "i_b"]


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
    from cicerone.policy.eligibility import _warned_missing_columns

    _warned_missing_columns.clear()
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
    # u3 has nationality=None; share that cohort with an unknown user.
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
    assert "empty allowlist" in caplog.text


def test_allowed_items_for_cohort_missing_items_with_user_rules_returns_empty(caplog):
    rules = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    allowed = allowed_items_for_cohort(["u1"], _users(), None, rules, ["i1", "i2"])
    assert allowed == []
    assert "empty allowlist" in caplog.text


def test_allowed_items_for_cohort_missing_items_with_item_only_rules_fails_open(caplog):
    rules = resolve_eligibility(_base_config())  # availability sugar only
    assert rules  # published + in_stock
    assert not has_user_scoped_eligibility(rules)
    allowed = allowed_items_for_cohort(["u1"], None, None, rules, ["i1", "i2"])
    assert allowed == ["i1", "i2"]
    assert "full catalog" in caplog.text


def test_allowed_items_for_cohort_empty_items_frame_matches_missing(caplog):
    """Empty items.parquet must fail-open for item-only rules (like items is None)."""
    rules = resolve_eligibility(_base_config())
    empty = pd.DataFrame(columns=["item_id", "published", "in_stock"])
    allowed = allowed_items_for_cohort(["u1"], None, empty, rules, ["i1", "i2"])
    assert allowed == ["i1", "i2"]
    assert "full catalog" in caplog.text

    user_rules = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    assert allowed_items_for_cohort(["u1"], _users(), empty, user_rules, ["i1", "i2"]) == []


def test_apply_boosts_truncates_even_when_items_missing():
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 3.0, "source": "popular_fallback"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 2.0, "source": "popular_fallback"},
            {"user_id": "u1", "item_id": "i3", "rank": 3, "score": 1.0, "source": "popular_fallback"},
        ]
    )
    boosts = [BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0)]
    out = apply_boosts(recs, None, boosts, top_k=2)
    assert list(out[Columns.Item]) == ["i1", "i2"]
    assert len(out) == 2


def test_item_boost_factors_warns_once_when_items_missing(caplog):
    import cicerone.policy.boosts as policy_boosts

    policy_boosts._warned_boost_without_items = False
    boosts = [BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0)]
    assert item_boost_factors(None, boosts) == {}
    assert item_boost_factors(pd.DataFrame(), boosts) == {}
    assert caplog.text.count("item boosts will not be applied") == 1


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
    assert out.iloc[0][BOOST_HITS_COLUMN] == [{"name": "paying", "factor": 2.0}]


def test_apply_boosts_omits_identity_factors():
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i2", "rank": 1, "score": 10.0, "source": "personalized"},
        ]
    )
    boosts = [
        BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0),
        BoostRule(
            name="tier",
            kind="value_map",
            item_column="plan_tier",
            value_factors={"premium": 1.5, "standard": 1.1, "free": 1.0},
        ),
    ]
    out = apply_boosts(recs, _items(), boosts, top_k=1)
    assert out.iloc[0][BOOST_HITS_COLUMN] == []


def test_as_list_handles_missing_series_and_zero_dim_ndarray():
    import numpy as np

    from cicerone.values import MISSING as _MISSING
    from cicerone.values import as_list as _as_list

    assert _as_list(None) == []
    assert _as_list(float("nan")) == []
    assert _as_list(_MISSING) == []
    assert _as_list(np.array("IT")) == ["IT"]
    assert _as_list(pd.Series(["beer", None, "wine"])) == ["beer", "wine"]
    assert _as_list(np.array(["IT", None], dtype=object)) == ["IT"]
    # Nested policy sentinel must be filtered like None / NaN.
    assert _as_list(["IT", _MISSING, "FR"]) == ["IT", "FR"]


def test_user_attr_missing_columns_and_empty_mask_edges():
    items = pd.DataFrame(
        [
            {"item_id": "i1", "available_countries": [], "category": None},
            {"item_id": "i2", "available_countries": [None], "category": float("nan")},
        ]
    )
    ships = [
        EligibilityRule(
            name="ships",
            op="user_in_item_list",
            item_column="available_countries",
            user_column="nationality",
        )
    ]
    # Empty / all-missing list cells → no matches (explode edges).
    assert not eligible_item_mask({"nationality": "IT"}, items, ships).any()
    only_empty = pd.DataFrame([{"item_id": "i1", "available_countries": []}])
    assert not eligible_item_mask({"nationality": "IT"}, only_empty, ships).any()

    cats = [
        EligibilityRule(
            name="cats",
            op="item_in_user_list",
            item_column="category",
            user_column="allowed_categories",
        )
    ]
    user_row = pd.Series({"user_id": "u1"})
    assert not eligible_item_mask(user_row, _items(), cats).any()
    assert not eligible_item_mask({"user_id": "u1"}, _items(), cats).any()


def test_numeric_factors_and_boost_without_top_k():
    from cicerone.policy.boosts import _numeric_factors
    from cicerone.values import is_missing as _is_missing

    items = pd.DataFrame([{"item_id": "i1", "margin": 1.0}])
    # Missing column / empty frame short-circuit inside helper.
    assert list(_numeric_factors(items, "missing", 1.0)) == [1.0]
    assert _numeric_factors(items.iloc[0:0], "margin", 1.0).empty
    # Non-scalar objects that pd.isna cannot truth-coerce.
    assert _is_missing({"nested": True}) is False

    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i2", "rank": 1, "score": 10.0},
            {"user_id": "u1", "item_id": "i1", "rank": 2, "score": 9.0},
        ]
    )
    boosts = [BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0)]
    out = apply_boosts(recs, _items(), boosts, top_k=None)
    assert list(out[Columns.Item]) == ["i1", "i2"]
    assert len(out) == 2


def test_eligible_item_mask_empty_inputs_and_unknown_op():
    items = _items()
    assert eligible_item_mask(None, items, []).all()
    empty_items = items.iloc[0:0]
    pub = [EligibilityRule(name="x", op="item_true", item_column="published")]
    assert eligible_item_mask(None, empty_items, pub).empty

    with pytest.raises(ValueError, match="Unknown eligibility op"):
        eligible_item_mask(
            {"market": "eu"},
            items,
            [EligibilityRule(name="bad", op="not_real", item_column="market", user_column="market")],
        )


def test_allowed_items_for_cohort_no_rules_returns_catalog():
    assert allowed_items_for_cohort(["u1"], None, _items(), [], ["i1", "i9"]) == ["i1", "i9"]


def test_boost_helpers_cover_missing_values_and_edge_numeric():
    items = pd.DataFrame(
        [
            {"item_id": "i1", "flag": None, "tier": None, "margin": 5.0},
            {"item_id": "i2", "flag": False, "tier": "free", "margin": 5.0},
            {"item_id": "i3", "flag": True, "tier": "premium", "margin": float("nan")},
        ]
    )
    factors = item_boost_factors(
        items,
        [
            BoostRule(name="flag", kind="boolean", item_column="flag", factor=2.0),
            BoostRule(
                name="tier",
                kind="value_map",
                item_column="tier",
                value_factors={"premium": 1.5},
            ),
            BoostRule(name="margin", kind="numeric", item_column="margin", weight=1.0),
        ],
    )
    # Missing boolean/value_map → 1.0; equal/NaN margins → normalized 0 → factor 1.0
    assert factors["i1"] == pytest.approx(1.0)
    assert factors["i2"] == pytest.approx(1.0)
    assert factors["i3"] == pytest.approx(2.0 * 1.5)

    empty_boost = [BoostRule(name="x", kind="boolean", item_column="flag", factor=2.0)]
    assert item_boost_factors(items.iloc[0:0], empty_boost) == {}
    with pytest.raises(ValueError, match="Unknown boost kind"):
        item_boost_factors(items, [BoostRule(name="bad", kind="nope", item_column="flag")])


def test_apply_boosts_empty_recs_and_no_boosts_paths():
    empty = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score])
    boosts = [BoostRule(name="paying", kind="boolean", item_column="is_paying_producer", factor=2.0)]
    assert apply_boosts(empty, _items(), boosts, top_k=1).empty

    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 3.0},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 2.0},
        ]
    )
    truncated = apply_boosts(recs, _items(), [], top_k=1)
    assert list(truncated[Columns.Item]) == ["i1"]
    untruncated = apply_boosts(recs, _items(), [], top_k=None)
    assert len(untruncated) == 2
    no_hits = apply_boosts(recs, _items(), boosts, top_k=2, record_hits=False)
    from cicerone.reasons import BOOST_HITS_COLUMN

    assert BOOST_HITS_COLUMN not in no_hits.columns


def test_boolean_boost_respects_string_false():
    items = pd.DataFrame(
        [
            {"item_id": "a", "featured": "true"},
            {"item_id": "b", "featured": "false"},
        ]
    )
    boosts = [BoostRule(name="f", kind="boolean", item_column="featured", factor=2.0)]
    factors = item_boost_factors(items, boosts)
    assert factors["a"] == 2.0
    assert factors["b"] == 1.0


def test_boolean_boost_respects_numeric_zero_and_nonzero():
    items = pd.DataFrame(
        [
            {"item_id": "z", "featured": 0},
            {"item_id": "one", "featured": 1},
            {"item_id": "two", "featured": 2},
            {"item_id": "zf", "featured": 0.0},
            {"item_id": "half", "featured": 0.5},
        ]
    )
    boosts = [BoostRule(name="f", kind="boolean", item_column="featured", factor=3.0)]
    factors = item_boost_factors(items, boosts)
    assert factors["z"] == 1.0
    assert factors["one"] == 3.0
    assert factors["two"] == 3.0
    assert factors["zf"] == 1.0
    assert factors["half"] == 3.0
