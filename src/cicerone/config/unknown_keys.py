"""Warn about config keys the loader would silently ignore (typos, removed options)."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

from cicerone.config.settings import (
    AutomlSettings,
    DashboardSettings,
    EvalSettings,
    EventsIncrementalSettings,
    EventsOnlineSettings,
    ExperimentSettings,
    ExplainSettings,
    PublishSettings,
    ServeSettings,
    TrackSettings,
    TriggerSettings,
    VariantSettings,
)

logger = logging.getLogger(__name__)


def _field_names(cls: type[Any], *extra: str) -> frozenset[str]:
    return frozenset(f.name for f in fields(cls)) | frozenset(extra)


_IO_KEYS = frozenset({"kind", "options", "artifact_hmac_key"})

# Free-form tables (``options``, ``[model.*]``, ``blending``, policy rules) are not checked.
KNOWN_KEYS: dict[str, frozenset[str]] = {
    "": frozenset(
        {"job", "input", "output", "serve", "dashboard", "events", "publish", "experiment", "track", "model"}
    ),
    "job": frozenset(
        {
            "feature_config_path",
            "top_k",
            "half_life_days",
            "cron_schedule",
            "mode",
            "models",
            "model_weights",
            "rrf_k",
            "save_model_artifact",
            "max_workers",
            "log_epoch_metrics",
            "epoch_metrics_every",
            "epoch_metrics_max_users",
            "epoch_metrics_regression_drop",
            "epoch_metrics_plateau_eps",
            "epoch_metrics_plateau_window",
            "item_based",
            "content_fallback",
            "sequential",
            "explain",
            "automl",
            "trigger",
            "eval",
        }
    ),
    "job.item_based": frozenset({"k_neighbors"}),
    "job.content_fallback": frozenset({"enabled", "max_neighbors"}),
    "job.sequential": frozenset({"min_median_interactions"}),
    "job.explain": _field_names(ExplainSettings),
    "job.automl": _field_names(AutomlSettings),
    "job.automl.candidates": frozenset({"models", "weights", "rrf_k"}),
    "job.trigger": _field_names(TriggerSettings),
    "job.eval": _field_names(EvalSettings),
    "input": _IO_KEYS,
    "output": _IO_KEYS,
    "serve": _field_names(ServeSettings),
    "dashboard": _field_names(DashboardSettings),
    "events": frozenset({"enabled", "kind", "options", "ha", "incremental", "online"}),
    "events.incremental": _field_names(EventsIncrementalSettings),
    "events.online": _field_names(EventsOnlineSettings),
    "publish": _field_names(PublishSettings),
    "experiment": _field_names(ExperimentSettings),
    "experiment.variants": _field_names(VariantSettings, "weights", "boost"),
    "track": _field_names(TrackSettings),
}


def warn_unknown_keys(raw: dict[str, Any]) -> list[str]:
    unknown: list[str] = []
    for path, allowed in KNOWN_KEYS.items():
        for prefix, table in _tables_at(raw, path):
            unknown.extend(f"{prefix}{key}" for key in table if key not in allowed)
    for key in unknown:
        logger.warning("Unknown config key %r is ignored", key)
    return unknown


def _tables_at(raw: dict[str, Any], path: str) -> list[tuple[str, dict[str, Any]]]:
    if not path:
        return [("", raw)]
    node: Any = raw
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict):
        return [(f"{path}.", node)]
    if isinstance(node, list):
        return [(f"{path}[{index}].", item) for index, item in enumerate(node) if isinstance(item, dict)]
    return []
