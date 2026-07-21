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


def test_load_feature_config_defaults_column_type_to_categorical(tmp_path):
    config_path = tmp_path / "no_type.toml"
    config_path.write_text('[[user_features]]\ncolumn = "region_slug"\n')

    config = load_feature_config(config_path)

    assert config.user_features[0].type == "categorical"


def test_load_feature_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_config(tmp_path / "does-not-exist.toml")
