from __future__ import annotations

import pandas as pd
import pytest
from rectools import Columns

from cicerone.config import STRATEGY_NAMES
from cicerone.dataset import build_dataset
from cicerone.model import DEFAULT_MODELS, STRATEGIES, train_and_recommend
from cicerone.model.strategies import as_recommender_model, validate_strategy_names


def test_strategies_keys_match_config_strategy_names():
    # Keep STRATEGIES in sync with config STRATEGY_NAMES.
    assert set(STRATEGIES) == set(STRATEGY_NAMES)


def test_validate_strategy_names_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="must match"):
        validate_strategy_names({"popular": STRATEGIES["popular"]}, ("popular", "latest"))


def test_as_recommender_model_accepts_every_registered_strategy():
    from cicerone.model import build_strategy_model
    from cicerone.model_config import SEQUENTIAL_STRATEGY, sequential_extra_available

    for name, strategy in STRATEGIES.items():
        if name == SEQUENTIAL_STRATEGY and not sequential_extra_available():
            continue
        model = strategy.factory() if strategy.factory is not None else build_strategy_model(name)
        assert as_recommender_model(model) is model, name


def test_build_sequential_without_extra_raises(monkeypatch):
    from cicerone.config import ConfigError
    from cicerone.model import build_strategy_model

    monkeypatch.setattr("cicerone.model.strategies.sequential_extra_available", lambda: False)
    with pytest.raises(ConfigError, match=r"cicerone-recommender\[sequential\]"):
        build_strategy_model("sequential")


def test_as_recommender_model_rejects_object_missing_recommend():
    class NotAModel:
        def fit(self, dataset):
            return self

    with pytest.raises(TypeError, match="does not implement the RecommenderModel protocol"):
        as_recommender_model(NotAModel())


def test_as_recommender_model_rejects_recommend_missing_expected_parameters():
    class WrongSignatureModel:
        def fit(self, dataset):
            return self

        def recommend(self, *, users, dataset, k):
            return pd.DataFrame()

    with pytest.raises(TypeError, match="missing expected parameter"):
        as_recommender_model(WrongSignatureModel())


def test_sequential_strategy_is_personalized_and_requires_interactions():
    assert STRATEGIES["sequential"].personalized is True
    assert STRATEGIES["sequential"].requires_interactions is True
    assert STRATEGIES["sequential"].source_label == "sequential"
    assert DEFAULT_MODELS == ["collaborative", "item_based", "popular"]


def test_item_based_and_content_fallback_require_interactions(feature_config, sample_users):
    """Feature-only warm users must not get item_based / content_fallback rows."""
    now = pd.Timestamp.now(tz="UTC")
    # u4 is feature-only in sample_users; give u1 beer interactions + a cold beer.
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {
                "item_id": "i_new",
                "category": "beer",
                "producer_id": "p9",
                "published": True,
                "in_stock": True,
            },
        ]
    )
    built = build_dataset(events, sample_users, items, feature_config, half_life_days=90)
    gated_models = ["item_based", "content_fallback"]
    gated_sources = {"item_based", "content_fallback"}

    # Full chain: feature-only u4 may get collaborative/popular, never gated sources.
    mixed = train_and_recommend(
        built,
        target_users=["u1", "u4"],
        config=feature_config,
        top_k=5,
        enabled_models=["collaborative", *gated_models, "popular"],
        content_fallback_enabled=True,
    )
    assert "u4" in set(mixed[Columns.User])
    feature_only = mixed[mixed[Columns.User] == "u4"]
    assert not feature_only["source"].isin(gated_sources).any()

    # Gated strategies alone: interacting user gets rows; feature-only gets none.
    interacting_only = train_and_recommend(
        built,
        target_users=["u1"],
        config=feature_config,
        top_k=5,
        enabled_models=gated_models,
        content_fallback_enabled=True,
    )
    assert not interacting_only.empty
    assert interacting_only["source"].isin(gated_sources).all()

    feature_only_gated = train_and_recommend(
        built,
        target_users=["u4"],
        config=feature_config,
        top_k=5,
        enabled_models=gated_models,
        content_fallback_enabled=True,
    )
    assert feature_only_gated.empty
