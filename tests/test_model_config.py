"""RecTools model_from_config / save-load tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from rectools import Columns
from rectools.models import (
    ImplicitItemKNNWrapperModel,
    LightFMWrapperModel,
    PopularModel,
    model_from_config,
)
from support.toml_config import write_toml

from cicerone.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    load_rectools_model,
    save_rectools_model,
)
from cicerone.config import ConfigError, load_settings
from cicerone.dataset import build_dataset
from cicerone.model import build_strategy_model
from cicerone.model_config import (
    DEFAULT_COLLABORATIVE_CONFIG,
    DEFAULT_ITEM_BASED_CONFIG,
    DEFAULT_LATEST_CONFIG,
    apply_legacy_item_based_k_neighbors,
    deep_merge,
    default_model_configs,
    item_based_k_from_config,
    resolve_model_configs,
)


def _minimal_io_toml() -> str:
    return """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
    """


def _tiny_dataset(feature_config):
    now = pd.Timestamp.now(tz="UTC")
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i3", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i3", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    return build_dataset(events, None, None, feature_config, half_life_days=90)


@pytest.mark.parametrize(
    ("strategy", "expected_cls"),
    [
        ("collaborative", LightFMWrapperModel),
        ("item_based", ImplicitItemKNNWrapperModel),
        ("popular", PopularModel),
        ("latest", PopularModel),
    ],
)
def test_default_toml_config_builds_expected_rectools_class(strategy, expected_cls):
    configs = default_model_configs()
    model = model_from_config(configs[strategy])
    assert isinstance(model, expected_cls)
    round_trip = model_from_config(model.get_config(simple_types=True))
    assert isinstance(round_trip, expected_cls)
    assert round_trip.get_params(simple_types=True) == model.get_params(simple_types=True)


def test_collaborative_and_item_based_hyperparameters_match_defaults():
    collab = model_from_config(DEFAULT_COLLABORATIVE_CONFIG)
    params = collab.get_params(simple_types=True)
    assert params["cls"] == "LightFMWrapperModel"
    assert params["epochs"] == 30
    assert params["model.no_components"] == 64
    assert params["model.loss"] == "warp"
    assert params["model.learning_rate"] == 0.05
    assert params["model.item_alpha"] == 1e-6
    assert params["model.user_alpha"] == 1e-6
    assert params["model.random_state"] == 42

    item = model_from_config(DEFAULT_ITEM_BASED_CONFIG)
    item_params = item.get_params(simple_types=True)
    assert item_params["cls"] == "ImplicitItemKNNWrapperModel"
    assert item_params["model.cls"] == "TFIDFRecommender"
    assert item_params["model.K"] == 20


def test_build_strategy_model_matches_model_from_config():
    configs = default_model_configs()
    for name in ("collaborative", "item_based"):
        via_helper = build_strategy_model(name, model_configs=configs, lightfm_num_threads=4)
        via_native = model_from_config(configs[name])
        assert via_helper.get_params(simple_types=True) == via_native.get_params(simple_types=True)


def test_legacy_k_neighbors_translates_to_model_K():
    configs = resolve_model_configs(
        legacy_k_neighbors=15,
        legacy_k_neighbors_explicit=True,
    )
    assert configs["item_based"]["model"]["K"] == 15
    model = model_from_config(configs["item_based"])
    assert model.get_params(simple_types=True)["model.K"] == 15


def test_native_model_item_based_K_from_toml(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [model.item_based]
        cls = "ImplicitItemKNNWrapperModel"
        [model.item_based.model]
        cls = "TFIDFRecommender"
        K = 11
        {_minimal_io_toml()}
        """,
    )
    settings = load_settings(config_path)
    assert settings.item_based_k_neighbors == 11
    assert settings.model_configs["item_based"]["model"]["K"] == 11
    model = model_from_config(settings.model_configs["item_based"])
    assert model.get_params(simple_types=True)["model.K"] == 11


def test_legacy_and_native_k_conflict_raises(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.item_based]
        k_neighbors = 9
        [model.item_based.model]
        K = 11
        {_minimal_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="Conflicting item_based neighbor"):
        load_settings(config_path)


def test_legacy_k_neighbors_still_loads(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.item_based]
        k_neighbors = 13
        {_minimal_io_toml()}
        """,
    )
    settings = load_settings(config_path)
    assert settings.item_based_k_neighbors == 13
    assert settings.model_configs["item_based"]["model"]["K"] == 13


def test_example_cicerone_toml_has_no_active_model_section():
    """Shipped example config relies on built-in defaults (no active [model.*])."""
    repo_config = Path(__file__).resolve().parents[1] / "config" / "cicerone.toml"
    for line in repo_config.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[model.") and not stripped.startswith("#"):
            pytest.fail(f"unexpected active model table in example config: {stripped}")
    defaults = default_model_configs()
    for name in ("collaborative", "item_based"):
        built = build_strategy_model(name, model_configs=defaults, lightfm_num_threads=4)
        assert built.get_params(simple_types=True) == model_from_config(defaults[name]).get_params(
            simple_types=True
        )


def test_rectools_save_load_model_recommend_round_trip(tmp_path, feature_config):
    built = _tiny_dataset(feature_config)
    model = build_strategy_model("popular")
    model.fit(built.dataset)
    users = list(built.dataset.user_id_map.external_ids)
    before = model.recommend(users=users, dataset=built.dataset, k=2, filter_viewed=False)

    path = tmp_path / "popular.rectools"
    save_rectools_model(path, model)
    loaded = load_rectools_model(path)
    after = loaded.recommend(users=users, dataset=built.dataset, k=2, filter_viewed=False)
    pd.testing.assert_frame_equal(
        before.sort_values([Columns.User, Columns.Rank]).reset_index(drop=True),
        after.sort_values([Columns.User, Columns.Rank]).reset_index(drop=True),
    )


def test_collaborative_save_load_model_recommend_round_trip(tmp_path, feature_config):
    built = _tiny_dataset(feature_config)
    model = build_strategy_model("collaborative", lightfm_num_threads=1)
    model.fit(built.dataset)
    users = list(built.dataset.user_id_map.external_ids)
    before = model.recommend(users=users, dataset=built.dataset, k=2, filter_viewed=False)

    path = tmp_path / "collaborative.rectools"
    save_rectools_model(path, model)
    loaded = load_rectools_model(path)
    after = loaded.recommend(users=users, dataset=built.dataset, k=2, filter_viewed=False)
    pd.testing.assert_frame_equal(
        before.sort_values([Columns.User, Columns.Rank]).reset_index(drop=True),
        after.sort_values([Columns.User, Columns.Rank]).reset_index(drop=True),
    )


def test_item_based_save_load_model_recommend_round_trip(tmp_path, feature_config):
    built = _tiny_dataset(feature_config)
    model = build_strategy_model("item_based")
    model.fit(built.dataset)
    users = list(built.dataset.user_id_map.external_ids)
    before = model.recommend(users=users, dataset=built.dataset, k=2, filter_viewed=True)

    path = tmp_path / "item_based.rectools"
    save_rectools_model(path, model)
    loaded = load_rectools_model(path)
    after = loaded.recommend(users=users, dataset=built.dataset, k=2, filter_viewed=True)
    pd.testing.assert_frame_equal(
        before.sort_values([Columns.User, Columns.Rank]).reset_index(drop=True),
        after.sort_values([Columns.User, Columns.Rank]).reset_index(drop=True),
    )


def test_legacy_and_native_k_same_value_ok(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [job.item_based]
        k_neighbors = 11
        [model.item_based.model]
        K = 11
        {_minimal_io_toml()}
        """,
    )
    settings = load_settings(config_path)
    assert settings.item_based_k_neighbors == 11


def test_unknown_model_table_raises(tmp_path):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [model.not_a_strategy]
        cls = "PopularModel"
        {_minimal_io_toml()}
        """,
    )
    with pytest.raises(ConfigError, match="unknown strategy"):
        load_settings(config_path)


def test_model_config_module_does_not_import_ml_stack():
    """Serve-safe: parsing model TOML must not pull rectools/lightfm/implicit/torch."""
    import cicerone.model_config as mc

    source = Path(mc.__file__).read_text()
    for name in ("rectools", "lightfm", "implicit", "torch", "pytorch_lightning"):
        assert f"import {name}" not in source
        assert f"from {name}" not in source


def test_artifact_schema_version_is_v3():
    assert ARTIFACT_SCHEMA_VERSION == 3


def test_item_based_and_latest_defaults_use_canonical_constants():
    from cicerone.config import DEFAULT_ITEM_BASED_K_NEIGHBORS
    from cicerone.model import LATEST_WINDOW_DAYS as MODEL_LATEST
    from cicerone.model_config import LATEST_WINDOW_DAYS as CONFIG_LATEST

    assert DEFAULT_ITEM_BASED_CONFIG["model"]["K"] == DEFAULT_ITEM_BASED_K_NEIGHBORS
    assert DEFAULT_LATEST_CONFIG["period"]["days"] == CONFIG_LATEST == MODEL_LATEST


def test_deep_merge_replaces_non_dict_leaves():
    merged = deep_merge({"a": {"b": 1}, "c": 2}, {"a": {"b": 9}, "c": {"d": 3}})
    assert merged == {"a": {"b": 9}, "c": {"d": 3}}


def test_item_based_k_from_config_missing_path_returns_none():
    assert item_based_k_from_config({}) is None
    assert item_based_k_from_config({"model": "not-a-dict"}) is None
    assert item_based_k_from_config({"model": {}}) is None


def test_legacy_k_neighbors_creates_model_dict_when_absent():
    configs = apply_legacy_item_based_k_neighbors(
        {"item_based": {"cls": "ImplicitItemKNNWrapperModel"}},
        k_neighbors=7,
        k_neighbors_explicit=True,
    )
    assert configs["item_based"]["model"]["K"] == 7


def test_resolve_model_configs_rejects_non_table_override():
    with pytest.raises(ConfigError, match="must be a table"):
        resolve_model_configs({"popular": "PopularModel"})


def test_resolve_model_configs_rejects_missing_cls(monkeypatch):
    monkeypatch.setattr(
        "cicerone.model_config.default_model_configs",
        lambda: {"popular": {"popularity": "n_interactions"}},
    )
    with pytest.raises(ConfigError, match="missing required key 'cls'"):
        resolve_model_configs({})
