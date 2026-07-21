"""Loads the user-editable feature/weight configuration (config/features.yml).

Kept as plain YAML instead of Python constants so event weights and which
user/item columns feed the model can change without touching code or
rebuilding the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("/app/config/features.yml")


@dataclass(frozen=True)
class FeatureColumn:
    column: str
    type: str  # "categorical" | "list"


@dataclass(frozen=True)
class FeatureConfig:
    event_weights: dict[str, float]
    quantity_scaled_events: set[str]
    event_caps: dict[str, int]
    user_features: list[FeatureColumn]
    item_features: list[FeatureColumn]
    item_availability_filters: list[str]


def load_feature_config(path: Path | str | None = None) -> FeatureConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def _columns(key: str) -> list[FeatureColumn]:
        return [
            FeatureColumn(column=c["column"], type=c.get("type", "categorical"))
            for c in raw.get(key, [])
        ]

    return FeatureConfig(
        event_weights={k: float(v) for k, v in raw.get("event_weights", {}).items()},
        quantity_scaled_events=set(raw.get("quantity_scaled_events", [])),
        event_caps={k: int(v) for k, v in raw.get("event_caps", {}).items()},
        user_features=_columns("user_features"),
        item_features=_columns("item_features"),
        item_availability_filters=list(raw.get("item_availability_filters", [])),
    )
