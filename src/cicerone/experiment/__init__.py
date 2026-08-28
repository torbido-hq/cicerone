"""Public experiment API."""

from __future__ import annotations

from cicerone.experiment.assignment import assign_variant, assignment_bucket, resolve_assignment
from cicerone.experiment.evaluate import (
    PRIMARY_METRIC_WEIGHTED,
    ExperimentReport,
    evaluate_experiment,
    exposure_row,
    user_outcome,
)
from cicerone.experiment.guardrails import GuardrailReport, evaluate_guardrails
from cicerone.experiment.recipes import (
    COMBINER_BLEND,
    COMBINER_PRIORITY,
    COMBINER_RRF,
    COMBINERS,
    CONTROL_NAME,
    TREATMENT_NAME,
    ResolvedRecipe,
    apply_recipe,
    inherit_combiner,
    recipes_manifest_json,
    resolve_recipes,
    union_models,
    validate_variant_models,
)
from cicerone.experiment.stats import ComparisonResult, VariantMetric, compare_variants, variant_metric
from cicerone.experiment.store import ExperimentStore, experiment_state

__all__ = [
    "COMBINER_BLEND",
    "COMBINER_PRIORITY",
    "COMBINER_RRF",
    "COMBINERS",
    "CONTROL_NAME",
    "ComparisonResult",
    "ExperimentReport",
    "ExperimentStore",
    "GuardrailReport",
    "PRIMARY_METRIC_WEIGHTED",
    "ResolvedRecipe",
    "TREATMENT_NAME",
    "VariantMetric",
    "apply_recipe",
    "assign_variant",
    "assignment_bucket",
    "compare_variants",
    "evaluate_experiment",
    "evaluate_guardrails",
    "experiment_state",
    "exposure_row",
    "inherit_combiner",
    "recipes_manifest_json",
    "resolve_assignment",
    "resolve_recipes",
    "union_models",
    "user_outcome",
    "validate_variant_models",
    "variant_metric",
]
