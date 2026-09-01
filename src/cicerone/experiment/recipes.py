"""Resolve per-variant ranking recipes from job defaults and experiment TOML."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, TypeVar, cast

from cicerone.config.constants import DEFAULT_MODELS, RRF_K, STRATEGY_NAMES, ConfigError
from cicerone.config.settings import Settings, VariantSettings
from cicerone.config.validation import validate_model_weights, validate_rrf_k
from cicerone.feature_config import (
    BLENDING_CURVES,
    BlendingConfig,
    BoostRule,
    EligibilityRule,
    FeatureConfig,
    parse_boost_rules,
    parse_eligibility_rules,
)

COMBINER_PRIORITY = "priority"
COMBINER_RRF = "rrf"
COMBINER_BLEND = "blend"
COMBINERS: tuple[str, ...] = (COMBINER_PRIORITY, COMBINER_RRF, COMBINER_BLEND)

CONTROL_NAME = "control"
TREATMENT_NAME = "treatment"


@dataclass(frozen=True)
class ResolvedRecipe:
    name: str
    traffic: float
    models: tuple[str, ...]
    weights: dict[str, float] | None
    rrf_k: float | None
    combiner: str
    blending: BlendingConfig
    boosts: tuple[BoostRule, ...]
    eligibility: tuple[EligibilityRule, ...]

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "traffic": self.traffic,
            "models": list(self.models),
            "weights": self.weights,
            "rrf_k": self.rrf_k,
            "combiner": self.combiner,
            "boosts": [asdict(rule) for rule in self.boosts],
            "eligibility": [asdict(rule) for rule in self.eligibility],
        }


def inherit_combiner(settings: Settings, feature_config: FeatureConfig) -> str:
    if feature_config.blending.enabled:
        return COMBINER_BLEND
    if settings.model_weights is not None:
        return COMBINER_RRF
    return COMBINER_PRIORITY


def default_models(settings: Settings) -> list[str]:
    return list(settings.models) if settings.models else list(DEFAULT_MODELS)


def apply_recipe(feature_config: FeatureConfig, recipe: ResolvedRecipe) -> FeatureConfig:
    return replace(
        feature_config,
        blending=recipe.blending,
        boosts=list(recipe.boosts),
        eligibility=list(recipe.eligibility),
    )


_T = TypeVar("_T")


def resolve_boost_policy(
    spec: object,
    inherited: Sequence[BoostRule],
    *,
    label: str,
) -> tuple[BoostRule, ...]:
    return _resolve_policy_spec(spec, inherited, parse_boost_rules, label=label, rule_type=BoostRule)


def resolve_eligibility_policy(
    spec: object,
    inherited: Sequence[EligibilityRule],
    *,
    label: str,
) -> tuple[EligibilityRule, ...]:
    return _resolve_policy_spec(
        spec, inherited, parse_eligibility_rules, label=label, rule_type=EligibilityRule
    )


def _unique_policy_names(names: Sequence[object], *, label: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            raise ConfigError(f"{label} rule name must be non-empty")
        if name in seen:
            raise ConfigError(f"{label} duplicate rule name {name!r}")
        seen.add(name)
        out.append(name)
    return tuple(out)


def _pick_named(inherited: Sequence[_T], names: Sequence[str], *, label: str) -> tuple[_T, ...]:
    by_name: dict[str, _T] = {}
    for rule in inherited:
        name = rule.name  # type: ignore[attr-defined]
        if name in by_name:
            raise ConfigError(f"{label} duplicate inherited rule name {name!r}")
        by_name[name] = rule
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ConfigError(f"{label} unknown rule name(s) {missing}")
    return tuple(by_name[name] for name in names)


def _resolve_policy_spec(
    spec: object,
    inherited: Sequence[_T],
    parse: Callable[[Sequence[Mapping[str, Any]]], Sequence[_T]],
    *,
    label: str,
    rule_type: type[_T],
) -> tuple[_T, ...]:
    if spec is True:
        return tuple(inherited)
    if spec is False:
        return ()
    if not isinstance(spec, (list, tuple)):
        raise ConfigError(f"{label} must be true, false, rule names, or rule tables")
    items = tuple(spec)
    if not items:
        return ()
    if all(isinstance(item, str) for item in items):
        names = _unique_policy_names(items, label=label)
        return _pick_named(inherited, names, label=label)
    if all(isinstance(item, Mapping) for item in items):
        try:
            return tuple(parse([dict(item) for item in items]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{label}: {exc}") from exc
    if all(isinstance(item, rule_type) for item in items):
        return cast(tuple[_T, ...], items)
    raise ConfigError(f"{label} must be true, false, rule names, or rule tables")


def union_models(recipes: Sequence[ResolvedRecipe]) -> list[str]:
    seen: list[str] = []
    for recipe in recipes:
        for name in recipe.models:
            if name not in seen:
                seen.append(name)
    return seen


def recipes_manifest_json(recipes: Sequence[ResolvedRecipe]) -> str:
    return json.dumps([recipe.manifest_dict() for recipe in recipes], separators=(",", ":"))


def parse_job_recipe_from_manifest(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    """Recover models / weights / rrf_k from a successful job manifest."""
    if not manifest or manifest.get("status") != "success":
        return None
    for row in _experiment_variant_rows(manifest.get("experiment_variants")):
        recipe = _recipe_from_variant_row(row)
        if recipe is not None:
            return recipe
    raw_models = str(manifest.get("models") or "").strip()
    if not raw_models:
        return None
    models = [part for part in raw_models.split(",") if part]
    weights = _parse_weights_csv(str(manifest.get("model_weights") or ""))
    rrf_k_raw = manifest.get("rrf_k")
    rrf_k = None if rrf_k_raw in (None, "") else float(str(rrf_k_raw))
    return {"models": models, "weights": weights, "rrf_k": rrf_k}


def resolve_recipes(
    settings: Settings,
    feature_config: FeatureConfig,
    *,
    automl_models: list[str] | None = None,
    automl_weights: dict[str, float] | None = None,
    automl_rrf_k: float | None = None,
    last_manifest: dict[str, Any] | None = None,
) -> tuple[ResolvedRecipe, ...]:
    experiment = settings.experiment
    if not experiment.enabled:
        return ()
    variants = list(experiment.variants)
    if experiment.automl_challenger:
        variants = _challenger_variants(
            variants,
            last_manifest=last_manifest,
            automl_models=automl_models,
            automl_weights=automl_weights,
            automl_rrf_k=automl_rrf_k,
            settings=settings,
        )
    if len(variants) < 2:
        raise ConfigError("experiment requires at least two variants (or automl_challenger with a prior run)")
    inherited = inherit_combiner(settings, feature_config)
    recipes = [
        _resolve_one(
            variant,
            settings=settings,
            feature_config=feature_config,
            inherited_combiner=inherited,
            automl_models=automl_models,
            automl_weights=automl_weights,
            automl_rrf_k=automl_rrf_k,
        )
        for variant in variants
    ]
    return tuple(recipes)


def _challenger_variants(
    variants: list[VariantSettings],
    *,
    last_manifest: dict[str, Any] | None,
    automl_models: list[str] | None,
    automl_weights: dict[str, float] | None,
    automl_rrf_k: float | None,
    settings: Settings,
) -> list[VariantSettings]:
    if automl_models is None:
        raise ConfigError("experiment.automl_challenger requires AutoML to select a treatment recipe")
    prior = parse_job_recipe_from_manifest(last_manifest)
    control_models = list(prior["models"]) if prior else default_models(settings)
    control_weights = prior["weights"] if prior else settings.model_weights
    control_rrf_k = prior["rrf_k"] if prior else settings.rrf_k
    by_name = {variant.name: variant for variant in variants}
    control = by_name.get(CONTROL_NAME) or VariantSettings(name=CONTROL_NAME, traffic=0.5)
    treatment = by_name.get(TREATMENT_NAME) or VariantSettings(name=TREATMENT_NAME, traffic=0.5)
    control_combiner = COMBINER_RRF if control_weights else control.combiner
    treatment_combiner = COMBINER_RRF if automl_weights is not None else treatment.combiner
    control = replace(
        control,
        models=control.models or control_models,
        model_weights=control.model_weights if control.model_weights is not None else control_weights,
        rrf_k=control.rrf_k if control.rrf_k is not None else control_rrf_k,
        combiner=control.combiner or control_combiner,
    )
    treatment = replace(
        treatment,
        models=treatment.models or list(automl_models),
        model_weights=treatment.model_weights if treatment.model_weights is not None else automl_weights,
        rrf_k=treatment.rrf_k if treatment.rrf_k is not None else automl_rrf_k,
        combiner=treatment.combiner or treatment_combiner,
    )
    return [control, treatment]


def _resolve_one(
    variant: VariantSettings,
    *,
    settings: Settings,
    feature_config: FeatureConfig,
    inherited_combiner: str,
    automl_models: list[str] | None,
    automl_weights: dict[str, float] | None,
    automl_rrf_k: float | None,
) -> ResolvedRecipe:
    models = list(variant.models) if variant.models else default_models(settings)
    if automl_models is not None and variant.models is None and not settings.experiment.automl_challenger:
        models = list(automl_models)
    combiner = variant.combiner or inherited_combiner
    weights = variant.model_weights
    if weights is None:
        if combiner == COMBINER_RRF:
            weights = (
                automl_weights
                if automl_weights is not None and variant.models is None
                else settings.model_weights
            )
            if weights is None:
                weights = {name: 1.0 for name in models}
        elif automl_weights is not None and variant.models is None and combiner != COMBINER_BLEND:
            weights = automl_weights
    if combiner == COMBINER_PRIORITY:
        weights = None
    rrf_k = variant.rrf_k
    if rrf_k is None:
        rrf_k = automl_rrf_k if automl_rrf_k is not None else settings.rrf_k
    if combiner != COMBINER_RRF:
        rrf_k = None if combiner == COMBINER_PRIORITY else (rrf_k if rrf_k is not None else RRF_K)
    blending = _recipe_blending(feature_config.blending, variant, combiner)
    validate_model_weights(weights, context=f"experiment.variants[{variant.name}].weights")
    validate_rrf_k(rrf_k, context=f"experiment.variants[{variant.name}].rrf_k")
    if combiner == COMBINER_RRF and weights is not None:
        unknown = [name for name in weights if name not in models]
        if unknown:
            raise ConfigError(
                f"experiment.variants[{variant.name}].weights key(s) {unknown} are not in models {models}"
            )
    return ResolvedRecipe(
        name=variant.name,
        traffic=variant.traffic,
        models=tuple(models),
        weights=dict(weights) if weights is not None else None,
        rrf_k=rrf_k,
        combiner=combiner,
        blending=blending,
        boosts=resolve_boost_policy(
            variant.boosts,
            feature_config.boosts,
            label=f"experiment.variants[{variant.name}].boosts",
        ),
        eligibility=resolve_eligibility_policy(
            variant.eligibility,
            feature_config.eligibility,
            label=f"experiment.variants[{variant.name}].eligibility",
        ),
    )


def _recipe_blending(
    base: BlendingConfig,
    variant: VariantSettings,
    combiner: str,
) -> BlendingConfig:
    enabled = combiner == COMBINER_BLEND
    overlay = variant.blending or {}
    updates: dict[str, Any] = {"enabled": enabled}
    if "curve" in overlay:
        curve = str(overlay["curve"])
        if curve not in BLENDING_CURVES:
            raise ConfigError(f"experiment.variants[{variant.name}].blending.curve {curve!r} is invalid")
        updates["curve"] = curve  # type: ignore[assignment]
    if "midpoint" in overlay:
        updates["midpoint"] = float(overlay["midpoint"])
    if "steepness" in overlay:
        updates["steepness"] = float(overlay["steepness"])
    if "saturate_at" in overlay:
        updates["saturate_at"] = float(overlay["saturate_at"])
    if "popular_share" in overlay:
        updates["popular_share"] = float(overlay["popular_share"])
    if "rrf_k" in overlay:
        updates["rrf_k"] = float(overlay["rrf_k"])
    if "latest_date_columns" in overlay:
        raw = overlay["latest_date_columns"]
        if isinstance(raw, str):
            updates["latest_date_columns"] = (raw,)
        else:
            updates["latest_date_columns"] = tuple(str(column) for column in raw)
    return replace(base, **updates)


def _experiment_variant_rows(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        parsed = raw
    else:
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    rows = [row for row in parsed if isinstance(row, dict)]
    if not rows:
        return []
    control = [row for row in rows if str(row.get("name") or "") == CONTROL_NAME]
    rest = [row for row in rows if str(row.get("name") or "") != CONTROL_NAME]
    return control + rest


def _recipe_from_variant_row(row: dict[str, Any]) -> dict[str, Any] | None:
    models_raw = row.get("models")
    if isinstance(models_raw, list):
        models = [str(part) for part in models_raw if part]
    elif isinstance(models_raw, str) and models_raw.strip():
        models = [part for part in models_raw.split(",") if part]
    else:
        return None
    if not models:
        return None
    weights = row.get("weights")
    if isinstance(weights, str):
        weights = _parse_weights_csv(weights)
    elif not isinstance(weights, dict):
        weights = None
    rrf_k_raw = row.get("rrf_k")
    rrf_k = None if rrf_k_raw in (None, "") else float(str(rrf_k_raw))
    return {"models": models, "weights": weights, "rrf_k": rrf_k}


def _parse_weights_csv(raw: str) -> dict[str, float] | None:
    if not raw.strip():
        return None
    weights: dict[str, float] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights or None


# Imported by config load for model-name checks without pulling recipes' ML defaults twice.
def validate_variant_models(models: list[str] | None, *, variant_name: str) -> None:
    if models is None:
        return
    if not models:
        raise ConfigError(f"experiment.variants[{variant_name}].models is empty")
    unknown = [name for name in models if name not in STRATEGY_NAMES]
    if unknown:
        raise ConfigError(
            f"experiment.variants[{variant_name}].models contains unknown model(s) {unknown}; "
            f"available: {list(STRATEGY_NAMES)}"
        )
