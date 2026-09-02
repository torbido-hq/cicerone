"""Redacted Settings view for the dashboard configuration page."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from cicerone import __version__
from cicerone.config import Settings
from cicerone.feature_config import load_feature_config

logger = logging.getLogger(__name__)

REDACTED = "[redacted]"
MISSING = "—"

_SECRET_KEYS = frozenset(
    {
        "auth_token",
        "metrics_token",
        "secret_access_key",
        "access_key_id",
        "aws_access_key_id",
        "database_url",
        "postgres_url",
        "redis_url",
        "endpoint_url",
        "queue_url",
        "password",
    }
)
_SECRET_KEY_RE = re.compile(
    r"(secret|password|token|auth|credential|api_key|private_key|url)",
    re.IGNORECASE,
)
_USERINFO_IN_URL_RE = re.compile(r"://[^/\s]+@")
_NESTED_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("model_configs", "Models", "[model.*]"),
    ("input", "Input", "[input]"),
    ("output", "Output", "[output]"),
    ("serve", "Serve", "[serve]"),
    ("dashboard", "Dashboard", "[dashboard]"),
    ("events", "Events", "[events]"),
    ("publish", "Publish", "[publish]"),
    ("track", "Track", "[track]"),
    ("eval", "Eval", "[job.eval]"),
    ("trigger", "Trigger", "[job.trigger]"),
    ("experiment", "Experiment", "[experiment]"),
)
_JOB_SKIP = frozenset({key for key, _title, _toml in _NESTED_SECTIONS} | {"mode"})

_DOCS = "https://cicerone.dev"
HINTS: dict[str, dict[str, str]] = {
    "meta": {
        "text": "Identity of this dashboard process, not the job or serve replica.",
        "docs": f"{_DOCS}/architecture/#dashboard",
    },
    "meta.mode": {
        "text": "batch or serve. The dashboard runs in either mode.",
        "docs": f"{_DOCS}/architecture/#dashboard",
    },
    "meta.version": {"text": "Installed cicerone package version."},
    "meta.config_path": {
        "text": "TOML this process opened. Job and serve may point at a different file.",
        "docs": f"{_DOCS}/architecture/#dashboard",
    },
    "job": {
        "text": "The batch train-and-write job: strategies, top-K, and schedule.",
        "docs": f"{_DOCS}/how-it-works/",
    },
    "job.top_k": {
        "text": "How many items the job writes for each user.",
        "docs": f"{_DOCS}/how-it-works/",
    },
    "job.models": {
        "text": "Which ranking strategies to fit this run (LightFM, KNN, optional EASE/ALS/sequential).",
        "docs": f"{_DOCS}/how-it-works/#strategies",
    },
    "job.model_weights": {
        "text": "When set, combine strategies with weighted reciprocal rank fusion.",
        "docs": f"{_DOCS}/how-it-works/#combining-strategies",
    },
    "job.rrf_k": {
        "text": "RRF smoothing constant when model_weights is set.",
        "docs": f"{_DOCS}/how-it-works/#combining-strategies",
    },
    "job.half_life_days": {
        "text": "Exponential recency decay on interaction timestamps.",
        "docs": f"{_DOCS}/how-it-works/#interaction-weighting",
    },
    "job.cron_schedule": {"text": "When the scheduler starts a full job run."},
    "job.feature_config_path": {
        "text": "Path to features.toml (weights, eligibility, boosts).",
        "docs": f"{_DOCS}/how-it-works/#interaction-weighting",
    },
    "job.save_model_artifact": {"text": "Write the fitted model next to the output store."},
    "job.max_workers": {"text": "Process pool size for strategy fit."},
    "job.content_fallback_enabled": {
        "text": "Recommend cold items from similar item metadata.",
        "docs": f"{_DOCS}/how-it-works/#strategies",
    },
    "job.sequential_min_median_interactions": {
        "text": "Skip sequential models when the typical user is too sparse.",
        "docs": f"{_DOCS}/how-it-works/#strategies",
    },
    "job.automl": {
        "text": "Time-split search over ranking recipes.",
        "docs": f"{_DOCS}/tutorial/#8-let-automl-pick-a-strategy-for-you",
    },
    "job.explain": {
        "text": "Persist a short why-this-item payload on each output row.",
        "docs": f"{_DOCS}/how-it-works/#why-this-item",
    },
    "model_configs": {
        "text": "Per-strategy hyperparameters under [model.<name>].",
        "docs": f"{_DOCS}/how-it-works/#strategies",
    },
    "input": {
        "text": "Where events, users, and items are read from.",
        "docs": f"{_DOCS}/architecture/",
    },
    "input.kind": {"text": "dataset files or a database."},
    "output": {
        "text": "Where recommendations and the run manifest are written.",
        "docs": f"{_DOCS}/architecture/",
    },
    "output.kind": {"text": "dataset files or a database."},
    "serve": {
        "text": "Read-only HTTP API over the precomputed top-K table.",
        "docs": f"{_DOCS}/openapi/",
    },
    "serve.enabled": {"text": "Whether this config starts the serve API."},
    "serve.auth_token": {"text": "Bearer token for GET /recommendations. Shown redacted."},
    "serve.default_k": {"text": "Default number of rows when the client omits limit."},
    "serve.metrics_enabled": {"text": "Expose Prometheus metrics on the serve process."},
    "dashboard": {
        "text": "This status UI: Basic Auth, lookup, experiments, and this page.",
        "docs": f"{_DOCS}/architecture/#dashboard",
    },
    "dashboard.enabled": {"text": "Whether this config starts the dashboard process."},
    "dashboard.users_path": {"text": "TOML of dashboard usernames to bcrypt hashes."},
    "dashboard.lookup_k": {"text": "Max recommendation rows shown in Inspect user."},
    "dashboard.lookup_events": {"text": "Max recent input events shown in Inspect user."},
    "events": {
        "text": "Incremental ingest between full retrains, plus optional online LightFM.",
        "docs": f"{_DOCS}/incremental-events/",
    },
    "events.enabled": {"text": "Turn on the events worker."},
    "events.kind": {"text": "webhook, db, s3, redis_streams, kafka, or rabbitmq."},
    "events.online": {
        "text": "Continue LightFM for users who sent events since the last full fit.",
        "docs": f"{_DOCS}/incremental-events/#online-collaborative-refresh",
    },
    "trigger": {
        "text": "HTTP webhook (and optional poll) to start a retrain.",
        "docs": f"{_DOCS}/tutorial/#14-trigger-a-retrain-on-demand",
    },
    "trigger.enabled": {"text": "Whether this config starts the trigger service."},
    "experiment": {
        "text": "Sticky A/B test of ranking recipes, with promote on this dashboard.",
        "docs": f"{_DOCS}/experiments/",
    },
    "experiment.enabled": {"text": "Assign users to variants at serve and job time."},
    "publish": {
        "text": "Emit per-user recommendation JSON after the output store write.",
        "docs": f"{_DOCS}/incremental-events/",
    },
    "publish.enabled": {"text": "Turn on the recommendation publisher sidecar."},
    "publish.kind": {"text": "kafka or rabbitmq. Serve still looks up from dataset/db."},
    "track": {
        "text": "Impression and click ingest for CTR/conversion, off the training event path.",
        "docs": f"{_DOCS}/evaluation/",
    },
    "track.enabled": {
        "text": (
            "Accept POST /track. GET /recommendations is not an impression "
            "unless serve.log_impressions is on."
        ),
    },
    "eval": {
        "text": "Production replay of the previous lists against later events (HitRate / NDCG / Recall).",
        "docs": f"{_DOCS}/evaluation/",
    },
    "eval.enabled": {"text": "Score the last written lists at the start of the next job."},
    "features": {
        "text": "Event weights, feature columns, eligibility, and boosts.",
        "docs": f"{_DOCS}/how-it-works/#interaction-weighting",
    },
    "features.event_weights": {
        "text": "Base weight per event_type before recency decay.",
        "docs": f"{_DOCS}/how-it-works/#interaction-weighting",
    },
    "features.eligibility": {
        "text": "Hard allowlists of which items a user may be recommended.",
        "docs": f"{_DOCS}/architecture/#business-policies",
    },
    "features.boosts": {
        "text": "Soft re-rank after strategies are combined.",
        "docs": f"{_DOCS}/architecture/#business-policies",
    },
}


def config_display(
    settings: Settings,
    *,
    config_path: str | None = None,
    usernames: tuple[str, ...] = (),
) -> dict[str, Any]:
    raw = asdict(settings)
    redacted = _redact(raw)
    job_fields = {
        field.name: _normalize(redacted.get(field.name))
        for field in fields(Settings)
        if field.name not in _JOB_SKIP
    }
    sections: list[dict[str, Any]] = [
        _decorate_section(
            {
                "id": "meta",
                "title": "Meta",
                "toml": None,
                "fields": {
                    "config_path": _normalize(config_path),
                    "version": __version__,
                    "mode": _normalize(redacted.get("mode")),
                },
                "message": None,
            }
        ),
        _decorate_section(
            {"id": "job", "title": "Job", "toml": "[job]", "fields": job_fields, "message": None}
        ),
    ]
    for key, title, toml in _NESTED_SECTIONS:
        section_fields = _normalize(redacted.get(key))
        if key == "dashboard" and isinstance(section_fields, dict):
            section_fields = {
                **section_fields,
                "users": _normalize(list(usernames)),
            }
        if not isinstance(section_fields, dict):
            section_fields = {"value": section_fields}
        sections.append(
            _decorate_section(
                {"id": key, "title": title, "toml": toml, "fields": section_fields, "message": None}
            )
        )
    feature_fields, feature_message = _feature_section(settings)
    sections.append(
        _decorate_section(
            {
                "id": "features",
                "title": "Features",
                "toml": "features.toml",
                "fields": feature_fields,
                "message": feature_message,
            }
        )
    )
    return {
        "sections": sections,
        "hints": HINTS,
        "split_note": (
            "This is the config this dashboard process loaded. Job and serve may use "
            "a different file when configs are split."
        ),
    }


def _decorate_section(section: dict[str, Any]) -> dict[str, Any]:
    fields = section.get("fields")
    badge: str | None = None
    kind: str | None = None
    if isinstance(fields, dict):
        enabled = fields.get("enabled")
        if enabled == "true":
            badge = "on"
        elif enabled == "false":
            badge = "off"
        raw_kind = fields.get("kind")
        if isinstance(raw_kind, str) and raw_kind not in {MISSING, REDACTED}:
            kind = raw_kind
        if section["id"] == "model_configs" and "value" not in fields:
            badge = f"{len(fields)} models"
    section["badge"] = badge
    section["kind"] = kind
    return section


def _is_secret_key(key: str, *, in_options: bool) -> bool:
    if key in _SECRET_KEYS:
        return True
    return in_options and _SECRET_KEY_RE.search(key) is not None


def _has_url_userinfo(value: Any) -> bool:
    return isinstance(value, str) and _USERINFO_IN_URL_RE.search(value) is not None


def _redact(value: Any, *, key: str | None = None, in_options: bool = False) -> Any:
    if key is not None and _is_secret_key(key, in_options=in_options):
        if value is None or value == "":
            return MISSING
        return REDACTED
    if _has_url_userinfo(value):
        return REDACTED
    if isinstance(value, dict):
        child_in_options = in_options or key == "options"
        return {
            str(child_key): _redact(child, key=child_key, in_options=child_in_options)
            for child_key, child in value.items()
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_redact(item, in_options=in_options) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_redact(item, in_options=in_options) for item in value]
    return value


def _normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value else MISSING
    if isinstance(value, list):
        items = [_normalize(item) for item in value]
        return items if items else MISSING
    if isinstance(value, dict):
        if not value:
            return MISSING
        return {str(child_key): _normalize(child) for child_key, child in value.items()}
    return value


def _feature_section(settings: Settings) -> tuple[Any, str | None]:
    path = Path(settings.feature_config_path)
    if not path.is_file():
        return None, f"No feature config file at {path}."
    try:
        loaded = load_feature_config(path)
    except Exception:
        logger.exception("Failed to load feature config for dashboard config page")
        return None, "Feature config could not be loaded."
    return _normalize(_redact(asdict(loaded))), None
