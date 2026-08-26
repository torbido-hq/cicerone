"""Dashboard experiment report + promote."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cicerone.config import Settings
from cicerone.events.store import load_recommendations_frame
from cicerone.experiment.evaluate import evaluate_experiment
from cicerone.experiment.recipes import ResolvedRecipe, resolve_recipes
from cicerone.experiment.store import ExperimentStore, experiment_state
from cicerone.feature_config import FeatureConfig, load_feature_config
from cicerone.io.factory import build_input_source, build_manifest_reader

logger = logging.getLogger(__name__)


def experiment_context(settings: Settings) -> dict[str, Any]:
    experiment = settings.experiment
    if not experiment.enabled:
        return {
            "enabled": False,
            "experiment": experiment,
            "report": None,
            "error": None,
            "promoted_variant": None,
        }
    store = ExperimentStore(settings.output)
    promoted = None
    try:
        state = store.read_state()
        if state and state.get("experiment_id") == experiment.id:
            promoted = state.get("promoted_variant")
            promoted = str(promoted) if promoted else None
    except Exception:
        logger.exception("Failed to read experiment state")
    feature_config = _load_features(settings)
    recipes = _recipes(settings, feature_config)
    if not recipes:
        return {
            "enabled": True,
            "experiment": experiment,
            "report": None,
            "error": "No experiment variants to evaluate.",
            "promoted_variant": promoted,
        }
    try:
        events = build_input_source(settings.input).read_events()
    except Exception:
        logger.exception("Failed to read events for experiment metrics")
        events = None
    if events is None:
        import pandas as pd

        events = pd.DataFrame()
    try:
        recs = load_recommendations_frame(settings.output)
    except Exception:
        logger.exception("Failed to load recommendations for experiment guardrails")
        recs = None
    try:
        exposures = store.read_exposures() if experiment.log_exposures else None
    except Exception:
        logger.exception("Failed to read experiment exposures")
        exposures = None
    catalog_size = None
    if recs is not None and not recs.empty and "item_id" in recs.columns:
        catalog_size = int(recs["item_id"].astype(str).nunique())
    weights = feature_config.event_weights if feature_config is not None else {}
    report = evaluate_experiment(
        experiment=experiment,
        recipes=recipes,
        events=events,
        event_weights=weights,
        recommendations=recs,
        exposures=exposures,
        promoted_variant=promoted,
        catalog_size=catalog_size,
    )
    return {
        "enabled": True,
        "experiment": experiment,
        "report": report,
        "error": None,
        "promoted_variant": promoted,
        "recipes": recipes,
    }


def promote_winner(settings: Settings, variant: str) -> str | None:
    context = experiment_context(settings)
    report = context.get("report")
    names = {item.name for item in settings.experiment.variants}
    if report is not None:
        names.update(item.treatment.name for item in report.comparisons)
        names.update(item.control.name for item in report.comparisons)
    if variant not in names:
        return f"Unknown variant {variant!r}"
    if report is None:
        return "Experiment report is not available"
    if not report.can_promote:
        return "Experiment is not ready to promote (" + ", ".join(report.promote_blocked_by) + ")"
    if report.winner and report.winner != variant:
        return f"Winner is {report.winner!r}, not {variant!r}"
    ExperimentStore(settings.output).write_state(
        experiment_state(settings.experiment.id, promoted_variant=variant)
    )
    return None


def _load_features(settings: Settings) -> FeatureConfig | None:
    path = Path(settings.feature_config_path)
    if not path.is_file():
        return None
    try:
        return load_feature_config(path)
    except Exception:
        logger.exception("Failed to load feature config for experiment page")
        return None


def _recipes(settings: Settings, feature_config: FeatureConfig | None) -> tuple[ResolvedRecipe, ...]:
    if feature_config is None:
        return ()
    try:
        last = build_manifest_reader(settings.output).read_latest()
    except Exception:
        last = None
        logger.exception("Failed to read manifest for experiment recipes")
    try:
        recipes = resolve_recipes(settings, feature_config, last_manifest=last)
        if recipes:
            return recipes
    except Exception:
        logger.exception("Failed to resolve experiment recipes from config")
    if last and last.get("experiment_variants"):
        try:
            parsed = json.loads(str(last["experiment_variants"]))
        except json.JSONDecodeError:
            return ()
        from cicerone.feature_config import BlendingConfig

        return tuple(
            ResolvedRecipe(
                name=str(item["name"]),
                traffic=float(item.get("traffic", 0.0)),
                models=tuple(item.get("models") or ()),
                weights=item.get("weights"),
                rrf_k=item.get("rrf_k"),
                combiner=str(item.get("combiner") or "priority"),
                blending=BlendingConfig(enabled=str(item.get("combiner")) == "blend"),
                boosts=bool(item.get("boosts", True)),
                eligibility=bool(item.get("eligibility", True)),
            )
            for item in parsed
        )
    return ()
