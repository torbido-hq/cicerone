from __future__ import annotations

import pandas as pd
from rectools import Columns
from support.model_events import synthetic_events

from cicerone.dataset import build_dataset
from cicerone.model import (
    STRATEGIES,
    RecommenderModel,
    Strategy,
    train_and_recommend,
)


def test_train_and_recommend_reuses_strategy_cache_across_calls(sample_items, feature_config, monkeypatch):
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    fit_calls = []
    from cicerone.model import build_strategy_model

    def counting_factory():
        model = build_strategy_model("popular")
        original_fit = model.fit

        def counting_fit(dataset):
            fit_calls.append(1)
            return original_fit(dataset)

        model.fit = counting_fit
        return model

    monkeypatch.setitem(
        STRATEGIES,
        "popular",
        Strategy(personalized=False, source_label="popular_fallback", factory=counting_factory),
    )

    cache: dict[str, RecommenderModel] = {}
    first = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular"],
        strategy_cache=cache,
    )
    second = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular"],
        strategy_cache=cache,
    )

    assert len(fit_calls) == 1
    assert "popular" in cache
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_train_and_recommend_strategy_cache_reused_across_different_top_k_and_weights(
    sample_items, feature_config, monkeypatch
):
    # Cache fitted models, not recommend() output — different top_k/weights reuse the fit.
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    fit_calls = []
    from cicerone.model import build_strategy_model

    def counting_factory():
        model = build_strategy_model("popular")
        original_fit = model.fit

        def counting_fit(dataset):
            fit_calls.append(1)
            return original_fit(dataset)

        model.fit = counting_fit
        return model

    monkeypatch.setitem(
        STRATEGIES,
        "popular",
        Strategy(personalized=False, source_label="popular_fallback", factory=counting_factory),
    )

    cache: dict[str, RecommenderModel] = {}
    small_top_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=1,
        enabled_models=["popular"],
        strategy_cache=cache,
    )
    large_top_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular"],
        weights={"popular": 2.0},
        strategy_cache=cache,
    )

    assert len(fit_calls) == 1
    assert (small_top_k.groupby(Columns.User).size() <= 1).all()
    assert (large_top_k.groupby(Columns.User).size() <= 5).all()


def test_train_and_recommend_without_cache_refits_every_call(sample_items, feature_config, monkeypatch):
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    fit_calls = []
    from cicerone.model import build_strategy_model

    def counting_factory():
        model = build_strategy_model("popular")
        original_fit = model.fit

        def counting_fit(dataset):
            fit_calls.append(1)
            return original_fit(dataset)

        model.fit = counting_fit
        return model

    monkeypatch.setitem(
        STRATEGIES,
        "popular",
        Strategy(personalized=False, source_label="popular_fallback", factory=counting_factory),
    )

    train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2, enabled_models=["popular"]
    )
    train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2, enabled_models=["popular"]
    )

    assert len(fit_calls) == 2


def test_train_and_recommend_parallel_fit_matches_sequential(sample_items, feature_config):
    # Deterministic strategies: parallel fit must match sequential.
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)
    kwargs = dict(
        built=built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular", "latest"],
    )

    sequential = train_and_recommend(**kwargs, max_workers=1)
    # max_workers > model count must be capped, not raise.
    parallel = train_and_recommend(**kwargs, max_workers=10)

    pd.testing.assert_frame_equal(sequential.reset_index(drop=True), parallel.reset_index(drop=True))


def test_train_and_recommend_parallel_fit_populates_strategy_cache(sample_items, feature_config):
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)
    cache: dict[str, RecommenderModel] = {}

    train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular", "latest"],
        strategy_cache=cache,
        max_workers=2,
    )

    assert set(cache) == {"popular", "latest"}


def test_resolve_recommend_models_inserts_content_fallback_before_popular():
    from cicerone.model import resolve_recommend_models

    assert resolve_recommend_models(
        ["collaborative", "item_based", "popular"],
        blending_enabled=False,
        content_fallback_enabled=True,
    ) == ["collaborative", "item_based", "content_fallback", "popular"]


def test_resolve_recommend_models_skips_content_fallback_when_disabled(caplog):
    from cicerone.model import resolve_recommend_models

    with caplog.at_level("INFO"):
        resolved = resolve_recommend_models(
            ["collaborative", "content_fallback", "popular"],
            blending_enabled=False,
            content_fallback_enabled=False,
        )
    assert resolved == ["collaborative", "popular"]
    assert "content_fallback is listed" in caplog.text


def test_resolve_run_models_centralizes_content_fallback_and_blending():
    from cicerone.model import content_fallback_enabled_from_models, plan_model_run, resolve_run_models

    resolved, recommend = resolve_run_models(
        ["collaborative", "popular"],
        blending_enabled=False,
        content_fallback_enabled=True,
    )
    assert resolved == ["collaborative", "popular"]
    assert recommend == ["collaborative", "content_fallback", "popular"]
    assert content_fallback_enabled_from_models(recommend) is True
    assert content_fallback_enabled_from_models(["collaborative", "popular"]) is False

    plan = plan_model_run(
        ["collaborative", "content_fallback", "popular"],
        blending_enabled=False,
        content_fallback_enabled=None,
    )
    assert plan.content_fallback_active is True
    assert list(plan.recommend_models) == ["collaborative", "content_fallback", "popular"]


def test_item_based_k_neighbors_reaches_tfidf_recommender(sample_items, feature_config):
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    fitted: dict = {}
    train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["item_based"],
        item_based_k_neighbors=7,
        strategy_cache=fitted,
    )
    params = fitted["item_based"].get_params(simple_types=True)
    assert params["model.K"] == 7
