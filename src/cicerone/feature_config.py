"""Loads the user-editable feature/weight configuration (config/features.toml).

Kept as plain TOML instead of Python constants so event weights and which
user/item columns feed the model can change without touching code or
rebuilding the image.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("/app/config/features.toml")

ELIGIBILITY_OPS = frozenset({"item_true", "eq", "user_in_item_list", "item_in_user_list"})
BOOST_KINDS = frozenset({"boolean", "value_map", "numeric"})
ON_MISSING_USER_VALUES = frozenset({"exclude", "allow"})
# Single source of truth for the default boost candidate over-fetch multiplier.
DEFAULT_BOOST_OVERFETCH_FACTOR = 3


@dataclass(frozen=True)
class FeatureColumn:
    column: str
    type: str  # "categorical" | "list"


@dataclass(frozen=True)
class EligibilityRule:
    """Hard filter: an item must pass every rule to be recommendable.

    Ops:
      item_true          — item[item_column] is truthy (no user column)
      eq                 — user[user_column] == item[item_column]
      user_in_item_list  — user[user_column] is in item[item_column] (list)
      item_in_user_list  — item[item_column] is in user[user_column] (list)
    """

    name: str
    op: str
    item_column: str
    user_column: str | None = None
    on_missing_user: str = "exclude"  # "exclude" | "allow"


@dataclass(frozen=True)
class BoostRule:
    """Soft re-rank: multiplies an item's score after strategy combine.

    Kinds:
      boolean   — if item[item_column] is truthy, multiply by `factor`
      value_map — look up item[item_column] in `value_factors` (default 1.0)
      numeric   — score *= 1 + weight * min-max-normalized(item[item_column])
    """

    name: str
    kind: str
    item_column: str
    factor: float = 1.0
    value_factors: dict[str, float] = field(default_factory=dict)
    weight: float = 0.0


@dataclass(frozen=True)
class FeatureConfig:
    event_weights: dict[str, float]
    quantity_scaled_events: set[str]
    event_caps: dict[str, int]
    user_features: list[FeatureColumn]
    item_features: list[FeatureColumn]
    item_availability_filters: list[str]
    eligibility: list[EligibilityRule] = field(default_factory=list)
    boosts: list[BoostRule] = field(default_factory=list)
    # When [[boost]] is set, over-fetch this many times top_k before re-ranking.
    boost_overfetch_factor: int = DEFAULT_BOOST_OVERFETCH_FACTOR


def _parse_boost_overfetch_factor(raw: Any) -> int:
    factor = DEFAULT_BOOST_OVERFETCH_FACTOR if raw is None else int(raw)
    if factor < 1:
        raise ValueError(f"boost_overfetch_factor must be >= 1, got {factor}")
    return factor


def _parse_eligibility(raw_rules: list[dict[str, Any]]) -> list[EligibilityRule]:
    """Parse explicit [[eligibility]] tables.

    ``item_availability_filters`` stays separate sugar and is merged at
    recommend time via ``policy.resolve_eligibility``.
    """
    rules: list[EligibilityRule] = []
    for raw in raw_rules:
        name = str(raw.get("name") or raw.get("item_column") or "unnamed")
        op = str(raw["op"])
        if op not in ELIGIBILITY_OPS:
            raise ValueError(
                f"Unknown eligibility op {op!r} in rule {name!r}; available: {sorted(ELIGIBILITY_OPS)}"
            )
        item_column = str(raw["item_column"])
        user_column = raw.get("user_column")
        if op != "item_true" and not user_column:
            raise ValueError(f"Eligibility rule {name!r} with op {op!r} requires user_column")
        on_missing_user = str(raw.get("on_missing_user", "exclude"))
        if on_missing_user not in ON_MISSING_USER_VALUES:
            raise ValueError(
                f"Eligibility rule {name!r} has invalid on_missing_user {on_missing_user!r}; "
                f"available: {sorted(ON_MISSING_USER_VALUES)}"
            )
        rules.append(
            EligibilityRule(
                name=name,
                op=op,
                item_column=item_column,
                user_column=str(user_column) if user_column is not None else None,
                on_missing_user=on_missing_user,
            )
        )
    return rules


def _parse_boosts(raw_boosts: list[dict[str, Any]]) -> list[BoostRule]:
    boosts: list[BoostRule] = []
    for raw in raw_boosts:
        name = str(raw.get("name") or raw.get("item_column") or "unnamed")
        kind = str(raw["kind"])
        if kind not in BOOST_KINDS:
            raise ValueError(
                f"Unknown boost kind {kind!r} in rule {name!r}; available: {sorted(BOOST_KINDS)}"
            )
        item_column = str(raw["item_column"])
        value_factors_raw = raw.get("value_factors") or {}
        if not isinstance(value_factors_raw, dict):
            raise ValueError(f"Boost rule {name!r} value_factors must be a table/dict")
        value_factors = {str(k): float(v) for k, v in value_factors_raw.items()}
        if kind == "value_map" and not value_factors:
            raise ValueError(f"Boost rule {name!r} with kind 'value_map' requires value_factors")
        if kind == "boolean" and "factor" not in raw:
            raise ValueError(f"Boost rule {name!r} with kind 'boolean' requires factor")
        if kind == "numeric" and "weight" not in raw:
            raise ValueError(f"Boost rule {name!r} with kind 'numeric' requires weight")
        boosts.append(
            BoostRule(
                name=name,
                kind=kind,
                item_column=item_column,
                factor=float(raw.get("factor", 1.0)),
                value_factors=value_factors,
                weight=float(raw.get("weight", 0.0)),
            )
        )
    return boosts


def load_feature_config(path: Path | str | None = None) -> FeatureConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    def _columns(key: str) -> list[FeatureColumn]:
        return [
            FeatureColumn(column=c["column"], type=c.get("type", "categorical")) for c in raw.get(key, [])
        ]

    return FeatureConfig(
        event_weights={k: float(v) for k, v in raw.get("event_weights", {}).items()},
        quantity_scaled_events=set(raw.get("quantity_scaled_events", [])),
        event_caps={k: int(v) for k, v in raw.get("event_caps", {}).items()},
        user_features=_columns("user_features"),
        item_features=_columns("item_features"),
        item_availability_filters=list(raw.get("item_availability_filters", [])),
        eligibility=_parse_eligibility(list(raw.get("eligibility", []))),
        boosts=_parse_boosts(list(raw.get("boost", []))),
        boost_overfetch_factor=_parse_boost_overfetch_factor(raw.get("boost_overfetch_factor")),
    )
