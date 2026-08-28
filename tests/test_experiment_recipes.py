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
    resolve_boost_policy,
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
    assert payload[1]["boosts"] == []
    assert payload[0]["boosts"][0]["name"] == "featured"


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


def test_automl_challenger_prefers_control_recipe_from_experiment_variants() -> None:
    settings = make_settings(
        models=["popular"],
        experiment=ExperimentSettings(enabled=True, id="auto", automl_challenger=True),
    )
    last = {
        "status": "success",
        "models": "collaborative,item_based,popular",
        "model_weights": "collaborative=2,popular=1",
        "experiment_variants": json.dumps(
            [
                {"name": "control", "models": ["popular"], "weights": None, "rrf_k": None},
                {
                    "name": "treatment",
                    "models": ["item_based", "popular"],
                    "weights": {"item_based": 1.0, "popular": 1.0},
                    "rrf_k": 50,
                },
            ]
        ),
    }
    recipes = resolve_recipes(
        settings,
        _features(),
        automl_models=["item_based", "popular"],
        automl_weights={"item_based": 1.0, "popular": 1.0},
        automl_rrf_k=50.0,
        last_manifest=last,
    )
    assert list(recipes[0].models) == ["popular"]
    assert recipes[0].weights is None


def test_parse_job_recipe_from_manifest_falls_back_when_variants_invalid() -> None:
    from cicerone.experiment.recipes import parse_job_recipe_from_manifest

    recipe = parse_job_recipe_from_manifest(
        {
            "status": "success",
            "models": "collaborative,popular",
            "model_weights": "collaborative=2,popular=1",
            "experiment_variants": "{not-json",
        }
    )
    assert recipe is not None
    assert recipe["models"] == ["collaborative", "popular"]
    assert recipe["weights"] == {"collaborative": 2.0, "popular": 1.0}


def test_parse_job_recipe_from_manifest_variant_row_shapes() -> None:
    from cicerone.experiment.recipes import parse_job_recipe_from_manifest

    assert parse_job_recipe_from_manifest(None) is None
    assert parse_job_recipe_from_manifest({"status": "failed", "models": "popular"}) is None
    empty_models = parse_job_recipe_from_manifest({"status": "success", "models": ""})
    assert empty_models is None
    csv_models = parse_job_recipe_from_manifest(
        {
            "status": "success",
            "experiment_variants": json.dumps(
                [
                    {
                        "name": "control",
                        "models": "popular,latest",
                        "weights": "popular=1,latest=2",
                        "rrf_k": "",
                    }
                ]
            ),
        }
    )
    assert csv_models is not None
    assert csv_models["models"] == ["popular", "latest"]
    assert csv_models["weights"] == {"popular": 1.0, "latest": 2.0}
    skipped_empty = parse_job_recipe_from_manifest(
        {
            "status": "success",
            "models": "collaborative",
            "experiment_variants": json.dumps([{"name": "control", "models": []}]),
        }
    )
    assert skipped_empty is not None
    assert skipped_empty["models"] == ["collaborative"]
    listed = parse_job_recipe_from_manifest(
        {
            "status": "success",
            "experiment_variants": [
                "skip",
                {"name": "control", "models": ["popular"], "weights": 1, "rrf_k": 40},
            ],
        }
    )
    assert listed is not None
    assert listed["models"] == ["popular"]
    assert listed["weights"] is None
    assert listed["rrf_k"] == 40.0
    assert parse_job_recipe_from_manifest({"status": "success", "experiment_variants": "null"}) is None


def test_automl_challenger_requires_automl_pick() -> None:
    settings = make_settings(
        experiment=ExperimentSettings(enabled=True, id="auto", automl_challenger=True),
    )
    with pytest.raises(ConfigError, match="AutoML"):
        resolve_recipes(settings, _features())


def test_resolve_recipes_named_and_replacement_policy() -> None:
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            variants=(
                VariantSettings(name="control", traffic=0.5, boosts=("featured",), eligibility=False),
                VariantSettings(
                    name="treatment",
                    traffic=0.5,
                    boosts=(
                        BoostRule(
                            name="new-arrivals",
                            kind="boolean",
                            item_column="is_new",
                            factor=1.4,
                        ),
                    ),
                    eligibility=(
                        EligibilityRule(
                            name="published",
                            op="item_true",
                            item_column="published",
                        ),
                    ),
                ),
            ),
        ),
    )
    recipes = resolve_recipes(settings, _features())
    control = apply_recipe(_features(), recipes[0])
    treatment = apply_recipe(_features(), recipes[1])
    assert [rule.name for rule in control.boosts] == ["featured"]
    assert control.eligibility == []
    assert [rule.name for rule in treatment.boosts] == ["new-arrivals"]
    assert treatment.boosts[0].factor == 1.4
    assert [rule.name for rule in treatment.eligibility] == ["published"]
    payload = json.loads(recipes_manifest_json(recipes))
    assert payload[0]["boosts"][0]["name"] == "featured"
    assert payload[1]["eligibility"][0]["item_column"] == "published"


def test_resolve_recipes_unknown_policy_name() -> None:
    settings = make_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5, boosts=("missing",)),
            ),
        ),
    )
    with pytest.raises(ConfigError, match="unknown rule name"):
        resolve_recipes(settings, _features())


def test_resolve_boost_policy_from_manifest_dicts() -> None:
    rules = resolve_boost_policy(
        [{"name": "x", "kind": "boolean", "item_column": "featured", "factor": 1.1}],
        _features().boosts,
        label="boosts",
    )
    assert rules[0].name == "x"
    assert rules[0].factor == 1.1


def test_resolve_boost_policy_rejects_duplicate_names() -> None:
    with pytest.raises(ConfigError, match="duplicate rule name"):
        resolve_boost_policy(("featured", "featured"), _features().boosts, label="boosts")


def test_resolve_boost_policy_edge_shapes() -> None:
    inherited = _features().boosts
    assert resolve_boost_policy([], inherited, label="boosts") == ()
    with pytest.raises(ConfigError, match="must be true, false"):
        resolve_boost_policy(1, inherited, label="boosts")
    with pytest.raises(ConfigError, match="non-empty"):
        resolve_boost_policy(("featured", "  "), inherited, label="boosts")
    with pytest.raises(ConfigError, match="must be true, false"):
        resolve_boost_policy(["featured", {"name": "x"}], inherited, label="boosts")
    with pytest.raises(ConfigError, match="boolean"):
        resolve_boost_policy(
            [{"name": "x", "kind": "boolean", "item_column": "featured"}],
            inherited,
            label="boosts",
        )
