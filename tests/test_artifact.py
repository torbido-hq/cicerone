from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from cicerone.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    build_artifact,
    dumps_artifact,
    load_artifact,
    loads_artifact,
    recommend_from_artifact,
    save_artifact,
)
from cicerone.dataset import build_dataset
from cicerone.model import fit_strategies, train_and_recommend


def test_artifact_round_trip_recommendations_match(
    tmp_path, feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    target_users = ["u1", "u2", "u3", "u4"]
    enabled = ["collaborative", "popular"]
    top_k = 3

    fitted: dict = {}
    original = train_and_recommend(
        built,
        target_users,
        feature_config,
        top_k=top_k,
        enabled_models=enabled,
        strategy_cache=fitted,
    )

    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=enabled,
        model_weights=None,
        rrf_k=None,
    )
    path = tmp_path / "model.artifact"
    save_artifact(path, artifact)

    loaded = load_artifact(path)
    assert loaded.schema_version == ARTIFACT_SCHEMA_VERSION
    assert list(loaded.models) == enabled

    reloaded = recommend_from_artifact(loaded, target_users, top_k=top_k)
    pd.testing.assert_frame_equal(
        original.sort_values(["user_id", "rank"]).reset_index(drop=True),
        reloaded.sort_values(["user_id", "rank"]).reset_index(drop=True),
    )


def test_artifact_round_trip_with_weighted_fusion(
    tmp_path, feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    target_users = ["u1", "u2"]
    enabled = ["collaborative", "popular"]
    weights = {"collaborative": 1.0, "popular": 0.5}
    top_k = 2

    fitted: dict = {}
    original = train_and_recommend(
        built,
        target_users,
        feature_config,
        top_k=top_k,
        enabled_models=enabled,
        weights=weights,
        rrf_k=40,
        strategy_cache=fitted,
    )
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=enabled,
        model_weights=weights,
        rrf_k=40,
    )
    save_artifact(tmp_path / "model.artifact", artifact)
    reloaded = recommend_from_artifact(load_artifact(tmp_path / "model.artifact"), target_users, top_k=top_k)
    pd.testing.assert_frame_equal(
        original.sort_values(["user_id", "rank"]).reset_index(drop=True),
        reloaded.sort_values(["user_id", "rank"]).reset_index(drop=True),
    )


def test_loads_artifact_rejects_wrong_schema_version(
    feature_config, sample_events, sample_users, sample_items
):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    _, fitted = fit_strategies(built, ["u1"], enabled_models=["popular"])
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=["popular"],
        model_weights=None,
        rrf_k=None,
    )
    bad = replace(artifact, schema_version=ARTIFACT_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="Unsupported artifact schema_version"):
        loads_artifact(dumps_artifact(bad))


def test_loads_artifact_rejects_non_artifact_payload():
    with pytest.raises(TypeError, match="ModelArtifact"):
        loads_artifact(__import__("pickle").dumps({"not": "an artifact"}))


def test_fit_strategies_populates_cache(feature_config, sample_events, sample_users, sample_items):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    cache: dict = {}
    enabled, fitted = fit_strategies(
        built, ["u1", "u2"], enabled_models=["popular", "latest"], strategy_cache=cache
    )
    assert enabled == ["popular", "latest"]
    assert set(fitted) == {"popular", "latest"}
    assert set(cache) == {"popular", "latest"}
