from __future__ import annotations

import pytest

from cicerone.feature_config import load_feature_config


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


def test_load_feature_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_config(tmp_path / "does-not-exist.toml")
