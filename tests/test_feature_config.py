from __future__ import annotations

import pytest

from cicerone.feature_config import DEFAULT_BOOST_OVERFETCH_FACTOR, load_feature_config


def test_load_feature_config_parses_all_sections(tmp_path):
    config_path = tmp_path / "features.toml"
    config_path.write_text(
        """
quantity_scaled_events = ["purchase"]
item_availability_filters = ["published", "in_stock"]

[event_weights]
purchase = 4.0
view = 0.3

[event_caps]
view = 5

[[user_features]]
column = "favorite_styles"
type = "list"

[[user_features]]
column = "region_slug"
type = "categorical"

[[item_features]]
column = "category"
type = "categorical"

[[eligibility]]
name = "ships_to_user"
op = "user_in_item_list"
user_column = "nationality"
item_column = "available_countries"

[[boost]]
name = "paying_producer"
kind = "boolean"
item_column = "is_paying_producer"
factor = 1.5

[[boost]]
name = "plan_tier"
kind = "value_map"
item_column = "plan_tier"
value_factors = { premium = 1.5, free = 1.0 }
"""
    )

    config = load_feature_config(config_path)

    assert config.event_weights == {"purchase": 4.0, "view": 0.3}
    assert config.quantity_scaled_events == {"purchase"}
    assert config.event_caps == {"view": 5}
    assert [c.column for c in config.user_features] == ["favorite_styles", "region_slug"]
    assert config.user_features[0].type == "list"
    assert config.user_features[1].type == "categorical"
    assert [c.column for c in config.item_features] == ["category"]
    assert config.item_availability_filters == ["published", "in_stock"]
    assert len(config.eligibility) == 1
    assert config.eligibility[0].name == "ships_to_user"
    assert config.eligibility[0].op == "user_in_item_list"
    assert config.eligibility[0].user_column == "nationality"
    assert config.eligibility[0].item_column == "available_countries"
    assert [b.name for b in config.boosts] == ["paying_producer", "plan_tier"]
    assert config.boosts[0].kind == "boolean"
    assert config.boosts[0].factor == 1.5
    assert config.boosts[1].value_factors == {"premium": 1.5, "free": 1.0}
    assert config.boost_overfetch_factor == DEFAULT_BOOST_OVERFETCH_FACTOR


def test_load_feature_config_defaults_to_empty_sections(tmp_path):
    config_path = tmp_path / "empty.toml"
    config_path.write_text("")

    config = load_feature_config(config_path)

    assert config.event_weights == {}
    assert config.quantity_scaled_events == set()
    assert config.event_caps == {}
    assert config.user_features == []
    assert config.item_features == []
    assert config.item_availability_filters == []
    assert config.eligibility == []
    assert config.boosts == []
    assert config.boost_overfetch_factor == DEFAULT_BOOST_OVERFETCH_FACTOR


def test_load_feature_config_parses_boost_overfetch_factor(tmp_path):
    config_path = tmp_path / "overfetch.toml"
    config_path.write_text("boost_overfetch_factor = 5\n")
    assert load_feature_config(config_path).boost_overfetch_factor == 5


def test_load_feature_config_rejects_invalid_boost_overfetch_factor(tmp_path):
    config_path = tmp_path / "bad_overfetch.toml"
    config_path.write_text("boost_overfetch_factor = 0\n")
    with pytest.raises(ValueError, match="boost_overfetch_factor"):
        load_feature_config(config_path)


def test_load_feature_config_defaults_column_type_to_categorical(tmp_path):
    config_path = tmp_path / "no_type.toml"
    config_path.write_text('[[user_features]]\ncolumn = "region_slug"\n')

    config = load_feature_config(config_path)

    assert config.user_features[0].type == "categorical"


def test_load_feature_config_rejects_unknown_eligibility_op(tmp_path):
    config_path = tmp_path / "bad_elig.toml"
    config_path.write_text(
        """
[[eligibility]]
name = "bad"
op = "not_a_real_op"
item_column = "x"
user_column = "y"
"""
    )
    with pytest.raises(ValueError, match="Unknown eligibility op"):
        load_feature_config(config_path)


def test_load_feature_config_rejects_boolean_boost_without_factor(tmp_path):
    config_path = tmp_path / "bad_boost.toml"
    config_path.write_text(
        """
[[boost]]
name = "paying"
kind = "boolean"
item_column = "is_paying_producer"
"""
    )
    with pytest.raises(ValueError, match="requires factor"):
        load_feature_config(config_path)


def test_load_feature_config_rejects_eligibility_without_user_column(tmp_path):
    config_path = tmp_path / "elig_no_user.toml"
    config_path.write_text(
        """
[[eligibility]]
name = "ships"
op = "user_in_item_list"
item_column = "available_countries"
"""
    )
    with pytest.raises(ValueError, match="requires user_column"):
        load_feature_config(config_path)


def test_load_feature_config_rejects_invalid_on_missing_user(tmp_path):
    config_path = tmp_path / "bad_missing.toml"
    config_path.write_text(
        """
[[eligibility]]
name = "ships"
op = "eq"
user_column = "market"
item_column = "market"
on_missing_user = "shrug"
"""
    )
    with pytest.raises(ValueError, match="on_missing_user"):
        load_feature_config(config_path)


def test_load_feature_config_rejects_unknown_and_invalid_boosts(tmp_path):
    unknown = tmp_path / "unknown_boost.toml"
    unknown.write_text(
        """
[[boost]]
name = "bad"
kind = "magic"
item_column = "x"
"""
    )
    with pytest.raises(ValueError, match="Unknown boost kind"):
        load_feature_config(unknown)

    bad_factors = tmp_path / "bad_factors.toml"
    bad_factors.write_text(
        """
[[boost]]
name = "tier"
kind = "value_map"
item_column = "plan_tier"
value_factors = ["not", "a", "table"]
"""
    )
    with pytest.raises(ValueError, match="value_factors must be a table"):
        load_feature_config(bad_factors)

    empty_map = tmp_path / "empty_map.toml"
    empty_map.write_text(
        """
[[boost]]
name = "tier"
kind = "value_map"
item_column = "plan_tier"
"""
    )
    with pytest.raises(ValueError, match="requires value_factors"):
        load_feature_config(empty_map)

    no_weight = tmp_path / "no_weight.toml"
    no_weight.write_text(
        """
[[boost]]
name = "margin"
kind = "numeric"
item_column = "margin"
"""
    )
    with pytest.raises(ValueError, match="requires weight"):
        load_feature_config(no_weight)


def test_load_feature_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_config(tmp_path / "does-not-exist.toml")
