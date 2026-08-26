from __future__ import annotations

import json

import pytest
from conftest import make_settings

from cicerone.config import ConfigError
from cicerone.config.settings import ExperimentSettings, VariantSettings
from cicerone.experiment.recipes import (
    COMBINER_BLEND,
    COMBINER_PRIORITY,
    COMBINER_RRF,
    CONTROL_NAME,
    TREATMENT_NAME,
    apply_recipe,
    inherit_combiner,
    recipes_manifest_json,
    resolve_recipes,
    union_models,
)
from cicerone.feature_config import BlendingConfig, BoostRule, EligibilityRule, FeatureConfig


def _features(*, blending: bool = False) -> FeatureConfig:
    return FeatureConfig(
        event_weights={"purchase": 1.0},
        quantity_scaled_events=set(),
        event_caps={},
        user_features=[],
        item_features=[],
        item_availability_filters=[],
        eligibility=[
            EligibilityRule(name="in_stock", op="item_true", item_column="in_stock"),
        ],
        boosts=[BoostRule(name="featured", kind="boolean", item_column="featured", factor=1.2)],
        blending=BlendingConfig(enabled=blending),
    )


def test_inherit_combiner_from_job_and_features() -> None:
    features = _features(blending=True)
    assert inherit_combiner(make_settings(), features) == COMBINER_BLEND
    assert inherit_combiner(make_settings(model_weights={"popular": 1.0}), _features()) == COMBINER_RRF
    assert inherit_combiner(make_settings(), _features()) == COMBINER_PRIORITY


def test_resolve_recipes_overrides_combiner_and_union() -> None:
    settings = make_settings(
        models=["collaborative", "popular"],
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(
                    name="treatment",
                    traffic=0.5,
                    models=["item_based", "popular", "latest"],
                    combiner="blend",
                    boosts=False,
                ),
            ),
        ),
    )
    recipes = resolve_recipes(settings, _features())
    assert [recipe.name for recipe in recipes] == ["control", "treatment"]
    assert recipes[0].combiner == COMBINER_PRIORITY
    assert recipes[0].models == ("collaborative", "popular")
    assert recipes[1].combiner == COMBINER_BLEND
    assert recipes[1].blending.enabled is True
    assert union_models(recipes) == ["collaborative", "popular", "item_based", "latest"]
    stripped = apply_recipe(_features(), recipes[1])
    assert stripped.boosts == []
    assert stripped.eligibility
    payload = json.loads(recipes_manifest_json(recipes))
    assert payload[1]["name"] == "treatment"
    assert payload[1]["boosts"] is False


def test_automl_challenger_uses_last_manifest_as_control() -> None:
    settings = make_settings(
        models=["popular"],
        experiment=ExperimentSettings(enabled=True, id="auto", automl_challenger=True),
    )
    last = {
        "status": "success",
        "models": "collaborative,popular",
        "model_weights": "collaborative=2,popular=1",
        "rrf_k": 40,
    }
    recipes = resolve_recipes(
        settings,
        _features(),
        automl_models=["item_based", "popular"],
        automl_weights={"item_based": 1.0, "popular": 1.0},
        automl_rrf_k=50.0,
        last_manifest=last,
    )
    assert [recipe.name for recipe in recipes] == [CONTROL_NAME, TREATMENT_NAME]
    assert list(recipes[0].models) == ["collaborative", "popular"]
    assert recipes[0].weights == {"collaborative": 2.0, "popular": 1.0}
    assert list(recipes[1].models) == ["item_based", "popular"]


def test_automl_challenger_requires_automl_pick() -> None:
    settings = make_settings(
        experiment=ExperimentSettings(enabled=True, id="auto", automl_challenger=True),
    )
    with pytest.raises(ConfigError, match="AutoML"):
        resolve_recipes(settings, _features())
