"""Dashboard experiment report + promote."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.config import Settings
from cicerone.config.constants import ATTRIBUTION_CLICK, ATTRIBUTION_IMPRESSION, TRACK_KIND_IMPRESSION
from cicerone.evaluation import conversion_event_types, user_track_outcomes
from cicerone.events.store import load_items_catalog_size, load_recommendation_guardrail_rows
from cicerone.experiment.evaluate import evaluate_experiment
from cicerone.experiment.recipes import ResolvedRecipe, resolve_recipes
from cicerone.experiment.store import ExperimentStore, experiment_state
from cicerone.feature_config import FeatureConfig, load_feature_config
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.factory import build_input_source, build_manifest_reader
from cicerone.io.options import is_s3_not_found, read_parquet, require_option, sql_identifier
from cicerone.io.recommendation_schema import USER_COLUMN
from cicerone.track.store import TrackStore

logger = logging.getLogger(__name__)

_EVENT_METRIC_COLUMNS = (USER_COLUMN, "item_id", "event_type", "quantity", "occurred_at")
_PROMOTE_STATE: dict[str, dict[str, Any]] = {}


def _track_rows_for_experiment(rows: Sequence[dict[str, Any]], experiment_id: str) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("experiment_id") or "")
        if row_id and row_id != experiment_id:
            continue
        matched.append(row)
    return matched


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
    promoted_at = None
    try:
        state = store.read_state()
        if state and str(state.get("experiment_id") or "") == str(experiment.id):
            _PROMOTE_STATE[experiment.id] = dict(state)
        else:
            _PROMOTE_STATE.pop(experiment.id, None)
    except Exception:
        logger.exception("Failed to read experiment state")
        state = store.last_state(experiment.id) or _PROMOTE_STATE.get(experiment.id)
    if state and str(state.get("experiment_id") or "") == str(experiment.id):
        promoted = state.get("promoted_variant")
        promoted = str(promoted) if promoted else None
        promoted_at = state.get("promoted_at")
        promoted_at = str(promoted_at) if promoted_at else None
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
        events = _load_metric_events(settings)
    except Exception:
        logger.exception("Failed to read events for experiment metrics")
        events = None
    if events is None:
        events = pd.DataFrame()
    try:
        recs = load_recommendation_guardrail_rows(settings.output)
    except Exception:
        logger.exception("Failed to load recommendations for experiment guardrails")
        recs = None
    try:
        exposures = store.read_exposures(experiment_id=experiment.id) if experiment.log_exposures else None
    except Exception:
        logger.exception("Failed to read experiment exposures")
        exposures = [] if experiment.log_exposures else None
    catalog_size = None
    try:
        catalog_size = load_items_catalog_size(settings.output)
    except Exception:
        logger.exception("Failed to read items snapshot for experiment catalog size")
    weights = feature_config.event_weights if feature_config is not None else {}
    track_outcomes = None
    n_impressions = 0
    if settings.track.enabled:
        try:
            track_rows = TrackStore(settings.output).read_rows()
        except Exception:
            logger.exception("Failed to read track rows for experiment metrics")
            track_rows = []
        track_rows = _track_rows_for_experiment(track_rows, experiment.id)
        n_impressions = sum(1 for row in track_rows if str(row.get("kind") or "") == TRACK_KIND_IMPRESSION)
        if experiment.attribution in {ATTRIBUTION_CLICK, ATTRIBUTION_IMPRESSION}:
            types = conversion_event_types(
                settings.track.conversion_event_types, primary_metric=experiment.primary_metric
            )
            conversions = events
            if not conversions.empty and "event_type" in conversions.columns:
                conversions = conversions[conversions["event_type"].astype(str).isin(set(types))]
            track_outcomes = user_track_outcomes(
                track_rows=track_rows,
                conversions=conversions,
                primary_metric=experiment.primary_metric,
                attribution=experiment.attribution,
                window_hours=settings.track.attribution_window_hours,
            )
    report = evaluate_experiment(
        experiment=experiment,
        recipes=recipes,
        events=events,
        event_weights=weights,
        recommendations=recs,
        exposures=exposures,
        promoted_variant=promoted,
        promoted_at=promoted_at,
        catalog_size=catalog_size,
        track_outcomes=track_outcomes,
        n_impressions=n_impressions,
        min_impressions=settings.track.min_impressions if settings.track.enabled else 0,
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
    payload = experiment_state(settings.experiment.id, promoted_variant=variant)
    ExperimentStore(settings.output).write_state(payload)
    _PROMOTE_STATE[settings.experiment.id] = dict(payload)
    return None


def clear_promotion(settings: Settings) -> str | None:
    if not settings.experiment.enabled:
        return "No experiment is enabled"
    payload = experiment_state(settings.experiment.id, promoted_variant=None)
    ExperimentStore(settings.output).write_state(payload)
    _PROMOTE_STATE[settings.experiment.id] = dict(payload)
    return None


def _load_metric_events(settings: Settings) -> pd.DataFrame:
    inp = settings.input
    if inp.kind == "dataset":
        try:
            frame = read_parquet(inp.options, "events.parquet", columns=list(_EVENT_METRIC_COLUMNS))
        except FileNotFoundError:
            return pd.DataFrame(columns=list(_EVENT_METRIC_COLUMNS))
        except Exception as exc:
            if is_s3_not_found(exc):
                return pd.DataFrame(columns=list(_EVENT_METRIC_COLUMNS))
            try:
                frame = read_parquet(inp.options, "events.parquet")
            except Exception:
                frame = build_input_source(inp).read_events()
        keep = [column for column in _EVENT_METRIC_COLUMNS if column in frame.columns]
        return frame.loc[:, keep] if keep else frame
    if inp.kind == "db" and not inp.options.get("events_query"):
        table = sql_identifier(
            inp.options.get("events_table", DEFAULT_EVENTS_TABLE),
            option="events_table",
        )
        engine = create_engine(require_option(inp.options, "database_url", "db"), pool_pre_ping=True)
        quoted = ", ".join(f'"{column}"' for column in _EVENT_METRIC_COLUMNS)
        try:
            return pd.read_sql(text(f'SELECT {quoted} FROM "{table}"'), engine)
        except Exception:
            return build_input_source(inp).read_events()
        finally:
            engine.dispose()
    frame = build_input_source(inp).read_events()
    keep = [column for column in _EVENT_METRIC_COLUMNS if column in frame.columns]
    return frame.loc[:, keep] if keep else frame


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
