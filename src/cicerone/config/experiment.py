"""``[experiment]`` settings coercion and TOML load helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from cicerone.config.constants import (
    ATTRIBUTION_CLICK,
    ATTRIBUTION_IMPRESSION,
    DEFAULT_EXPERIMENT_ALPHA,
    EXPERIMENT_ATTRIBUTIONS,
    EXPERIMENT_COMBINERS,
    PRIMARY_METRIC_CONVERSION,
    PRIMARY_METRIC_CTR,
    PRIMARY_METRIC_WEIGHTED,
    ConfigError,
)
from cicerone.config.settings import ExperimentSettings, VariantSettings
from cicerone.config.validation import require_open_unit_interval, validate_model_weights, validate_rrf_k
from cicerone.feature_config import parse_boost_rules, parse_eligibility_rules

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_POLICY_MISSING = object()


def _coerce_experiment(value: Any) -> ExperimentSettings:
    if value is None:
        return ExperimentSettings()
    if isinstance(value, ExperimentSettings):
        return value
    if isinstance(value, dict):
        return load_experiment_settings(value)
    raise TypeError(f"Expected ExperimentSettings, dict, or None; got {type(value).__name__}")


def load_experiment_settings(raw: dict[str, Any] | None) -> ExperimentSettings:
    data = raw or {}
    enabled = bool(data.get("enabled", False))
    variants_raw = data.get("variants") or []
    variants = tuple(_load_variant(item, index) for index, item in enumerate(variants_raw))
    experiment_id = str(data.get("id") or "").strip()
    default_metric = PRIMARY_METRIC_WEIGHTED
    primary_metric = str(data.get("primary_metric") or default_metric).strip() or default_metric
    alpha = float(data.get("alpha", DEFAULT_EXPERIMENT_ALPHA))
    automl_challenger = bool(data.get("automl_challenger", False))
    attribution = str(data.get("attribution") or "user").strip().lower() or "user"
    if attribution not in EXPERIMENT_ATTRIBUTIONS:
        raise ConfigError(
            f"experiment.attribution must be one of {list(EXPERIMENT_ATTRIBUTIONS)}, got {attribution!r}"
        )
    if primary_metric in {PRIMARY_METRIC_CTR, PRIMARY_METRIC_CONVERSION} and attribution not in {
        ATTRIBUTION_CLICK,
        ATTRIBUTION_IMPRESSION,
    }:
        raise ConfigError(
            "experiment.primary_metric 'ctr'/'conversion' requires attribution "
            "'click' or 'impression' (user keeps event ITT; recommended joins the list)"
        )
    if attribution in {ATTRIBUTION_CLICK, ATTRIBUTION_IMPRESSION} and primary_metric not in {
        PRIMARY_METRIC_CTR,
        PRIMARY_METRIC_CONVERSION,
    }:
        raise ConfigError(
            "experiment.attribution 'click'/'impression' requires primary_metric 'ctr' or 'conversion'"
        )
    if enabled:
        if not experiment_id:
            raise ConfigError("experiment.id is required when experiment.enabled = true")
        if not automl_challenger and len(variants) < 2:
            raise ConfigError("experiment requires at least two [[experiment.variants]] tables")
        if automl_challenger and variants and len(variants) < 2:
            raise ConfigError("experiment.automl_challenger with variants still needs at least two variants")
        names = [variant.name for variant in variants]
        if len(names) != len(set(names)):
            raise ConfigError(f"experiment.variants names must be unique, got {names}")
        variants = _normalize_traffic(variants)
        require_open_unit_interval(alpha, name="experiment.alpha")
        if not primary_metric:
            raise ConfigError("experiment.primary_metric must be a non-empty string")
    return ExperimentSettings(
        enabled=enabled,
        id=experiment_id,
        primary_metric=primary_metric,
        variants=variants,
        log_exposures=bool(data.get("log_exposures", False)),
        automl_challenger=automl_challenger,
        alpha=alpha,
        attribution=attribution,
    )


def _load_variant(raw: Any, index: int) -> VariantSettings:
    if not isinstance(raw, dict):
        raise ConfigError(f"experiment.variants[{index}] must be a table")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ConfigError(f"experiment.variants[{index}].name is required")
    traffic = float(raw.get("traffic", 0.0))
    if traffic < 0:
        raise ConfigError(f"experiment.variants[{name}].traffic must be >= 0, got {traffic}")
    models = list(raw["models"]) if "models" in raw else None
    if models is not None:
        from cicerone.experiment.recipes import validate_variant_models

        validate_variant_models(models, variant_name=name)
    weights = {str(key): float(value) for key, value in raw["weights"].items()} if "weights" in raw else None
    if weights is None and "model_weights" in raw:
        weights = {str(key): float(value) for key, value in raw["model_weights"].items()}
    validate_model_weights(weights, context=f"experiment.variants[{name}].weights")
    rrf_k = float(raw["rrf_k"]) if "rrf_k" in raw else None
    validate_rrf_k(rrf_k, context=f"experiment.variants[{name}].rrf_k")
    combiner = str(raw["combiner"]).lower() if "combiner" in raw else None
    if combiner is not None and combiner not in EXPERIMENT_COMBINERS:
        raise ConfigError(
            f"experiment.variants[{name}].combiner must be one of {list(EXPERIMENT_COMBINERS)}, "
            f"got {combiner!r}"
        )
    blending = dict(raw["blending"]) if isinstance(raw.get("blending"), dict) else None
    return VariantSettings(
        name=name,
        traffic=traffic,
        models=models,
        model_weights=weights,
        rrf_k=rrf_k,
        combiner=combiner,
        blending=blending,
        boosts=_load_policy_spec(
            raw, name, field="boosts", extra_field="boost", parse_table=parse_boost_rules
        ),
        eligibility=_load_policy_spec(
            raw, name, field="eligibility", extra_field=None, parse_table=parse_eligibility_rules
        ),
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


def _load_policy_spec(
    raw: dict[str, Any],
    name: str,
    *,
    field: str,
    extra_field: str | None,
    parse_table: Callable[[Sequence[Mapping[str, Any]]], Sequence[_T]],
) -> bool | tuple[str, ...] | tuple[_T, ...]:
    field_value = raw.get(field, _POLICY_MISSING)
    extra_value = raw.get(extra_field, _POLICY_MISSING) if extra_field else _POLICY_MISSING
    if field_value is not _POLICY_MISSING and extra_value is not _POLICY_MISSING:
        raise ConfigError(
            f"experiment.variants[{name}] must not set both {field} and [[experiment.variants.{extra_field}]]"
        )
    value = extra_value if extra_value is not _POLICY_MISSING else field_value
    if value is _POLICY_MISSING:
        return True
    if isinstance(value, bool):
        return value
    if not isinstance(value, list):
        raise ConfigError(
            f"experiment.variants[{name}].{field} must be true, false, "
            "a list of rule names, or an array of rule tables"
        )
    if not value:
        return ()
    label = f"experiment.variants[{name}].{field}"
    if all(isinstance(item, str) for item in value):
        return _unique_policy_names(value, label=label)
    if not all(isinstance(item, dict) for item in value):
        raise ConfigError(f"{label} must be true, false, a list of rule names, or an array of rule tables")
    try:
        rules = tuple(parse_table([dict(item) for item in value]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{label}: {exc}") from exc
    _unique_policy_names([rule.name for rule in rules], label=label)  # type: ignore[attr-defined]
    return rules


def _normalize_traffic(variants: tuple[VariantSettings, ...]) -> tuple[VariantSettings, ...]:
    if not variants:
        return variants
    total = sum(variant.traffic for variant in variants)
    if total > 1.0 + 1e-9:
        raise ConfigError(f"experiment.variants traffic sums to {total}, which exceeds 1")
    remainder = max(0.0, 1.0 - total)
    last = variants[-1]
    if remainder > 1e-9:
        logger.warning(
            "experiment.variants traffic sums to %s; assigning remainder %s to %r",
            total,
            remainder,
            last.name,
        )
    adjusted = replace(last, traffic=last.traffic + remainder)
    return (*variants[:-1], adjusted)
