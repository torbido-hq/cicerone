"""Redacted Settings view for the dashboard configuration page."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, fields
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
        "database_url",
        "postgres_url",
        "redis_url",
        "password",
    }
)
_SECRET_KEY_RE = re.compile(
    r"(secret|password|token|auth|credential|api_key|private_key)",
    re.IGNORECASE,
)
_NESTED_SECTIONS: tuple[tuple[str, str], ...] = (
    ("model_configs", "Models"),
    ("input", "Input"),
    ("output", "Output"),
    ("serve", "Serve"),
    ("dashboard", "Dashboard"),
    ("events", "Events"),
    ("trigger", "Trigger"),
    ("experiment", "Experiment"),
)
_JOB_SKIP = frozenset({key for key, _title in _NESTED_SECTIONS} | {"mode"})


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
        {
            "id": "meta",
            "title": "Meta",
            "fields": {
                "config_path": _normalize(config_path),
                "version": __version__,
                "mode": _normalize(redacted.get("mode")),
            },
            "message": None,
        },
        {"id": "job", "title": "Job", "fields": job_fields, "message": None},
    ]
    for key, title in _NESTED_SECTIONS:
        section_fields = _normalize(redacted.get(key))
        if key == "dashboard" and isinstance(section_fields, dict):
            section_fields = {
                **section_fields,
                "users": _normalize(list(usernames)),
            }
        if not isinstance(section_fields, dict):
            section_fields = {"value": section_fields}
        sections.append({"id": key, "title": title, "fields": section_fields, "message": None})
    feature_fields, feature_message = _feature_section(settings)
    sections.append(
        {
            "id": "features",
            "title": "Features",
            "fields": feature_fields,
            "message": feature_message,
        }
    )
    return {
        "sections": sections,
        "split_note": (
            "This is the config this dashboard process loaded. Job and serve may use "
            "a different file when configs are split."
        ),
    }


def _is_secret_key(key: str, *, in_options: bool) -> bool:
    if key in _SECRET_KEYS:
        return True
    return in_options and _SECRET_KEY_RE.search(key) is not None


def _redact(value: Any, *, key: str | None = None, in_options: bool = False) -> Any:
    if key is not None and _is_secret_key(key, in_options=in_options):
        if value is None or value == "":
            return MISSING
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
