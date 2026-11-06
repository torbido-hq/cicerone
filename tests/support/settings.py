from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from cicerone.config import (
    AutomlSettings,
    DashboardSettings,
    EvalSettings,
    IOSettings,
    ServeSettings,
    Settings,
    TrackSettings,
    TriggerSettings,
)
from cicerone.config import make_settings as _make_settings

_FLAT_SECTIONS: dict[str, type[Any]] = {
    "serve": ServeSettings,
    "trigger": TriggerSettings,
    "dashboard": DashboardSettings,
    "automl": AutomlSettings,
    "track": TrackSettings,
    "eval": EvalSettings,
}


def make_settings(**overrides: Any) -> Settings:
    """``cicerone.config.make_settings`` that also accepts flat ``<section>_<field>`` overrides."""
    for section, cls in _FLAT_SECTIONS.items():
        flat = {
            f.name: overrides.pop(f"{section}_{f.name}")
            for f in fields(cls)
            if f"{section}_{f.name}" in overrides
        }
        if not flat:
            continue
        nested = overrides.get(section)
        if nested is None or isinstance(nested, cls):
            overrides[section] = replace(cls() if nested is None else nested, **flat)
        elif isinstance(nested, dict):
            # Keep dict sections on the production path (e.g. load_track_settings validation).
            overrides[section] = {**nested, **flat}
    return _make_settings(**overrides)


def local_output(path: Any) -> IOSettings:
    return IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(path)})


def sqlite_output(path: Any) -> IOSettings:
    return IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{path}"})
