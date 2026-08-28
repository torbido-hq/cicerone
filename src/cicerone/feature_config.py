"""Loads config/features.toml (event weights and feature columns)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

DEFAULT_CONFIG_PATH = Path("/app/config/features.toml")

ELIGIBILITY_OPS = frozenset({"item_true", "eq", "user_in_item_list", "item_in_user_list"})
BOOST_KINDS = frozenset({"boolean", "value_map", "numeric"})
ON_MISSING_USER_VALUES = frozenset({"exclude", "allow"})
FEATURE_COLUMN_TYPES = frozenset({"categorical", "list"})
BLENDING_CURVES = frozenset({"sigmoid", "linear"})
DEFAULT_BOOST_OVERFETCH_FACTOR = 3
DEFAULT_BLENDING_MIDPOINT = 5.0
DEFAULT_BLENDING_STEEPNESS = 1.0
DEFAULT_BLENDING_SATURATE_AT = 10.0
DEFAULT_BLENDING_POPULAR_SHARE = 0.7
DEFAULT_BLENDING_RRF_K = 60.0
DEFAULT_LATEST_DATE_COLUMNS: tuple[str, ...] = ("published_at", "created_at", "occurred_at")

FeatureColumnType = Literal["categorical", "list"]
EligibilityOp = Literal["item_true", "eq", "user_in_item_list", "item_in_user_list"]
BoostKind = Literal["boolean", "value_map", "numeric"]
OnMissingUser = Literal["exclude", "allow"]
BlendingCurve = Literal["sigmoid", "linear"]


@dataclass(frozen=True)
class FeatureColumn:
    column: str
    type: FeatureColumnType


@dataclass(frozen=True)
class EligibilityRule:
    name: str
    op: EligibilityOp
    item_column: str
    user_column: str | None = None
    on_missing_user: OnMissingUser = "exclude"


@dataclass(frozen=True)
class BoostRule:
    name: str
    kind: BoostKind
    item_column: str
    factor: float = 1.0
    value_factors: dict[str, float] = field(default_factory=dict)
    weight: float = 0.0


@dataclass(frozen=True)
class BlendingConfig:
    """Per-user weighted blend of personalized / popular / latest sources."""

    enabled: bool = False
    curve: BlendingCurve = "sigmoid"
    midpoint: float = DEFAULT_BLENDING_MIDPOINT
    steepness: float = DEFAULT_BLENDING_STEEPNESS
    saturate_at: float = DEFAULT_BLENDING_SATURATE_AT
    popular_share: float = DEFAULT_BLENDING_POPULAR_SHARE
    latest_date_columns: tuple[str, ...] = DEFAULT_LATEST_DATE_COLUMNS
    rrf_k: float = DEFAULT_BLENDING_RRF_K


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
    boost_overfetch_factor: int = DEFAULT_BOOST_OVERFETCH_FACTOR
    blending: BlendingConfig = field(default_factory=BlendingConfig)


def _parse_boost_overfetch_factor(raw: Any) -> int:
    factor = DEFAULT_BOOST_OVERFETCH_FACTOR if raw is None else int(raw)
    if factor < 1:
        raise ValueError(f"boost_overfetch_factor must be >= 1, got {factor}")
    return factor


def parse_eligibility_rules(raw_rules: list[dict[str, Any]]) -> list[EligibilityRule]:
    return _parse_eligibility(raw_rules)


def parse_boost_rules(raw_boosts: list[dict[str, Any]]) -> list[BoostRule]:
    return _parse_boosts(raw_boosts)


def _parse_eligibility(raw_rules: list[dict[str, Any]]) -> list[EligibilityRule]:
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
                op=cast(EligibilityOp, op),
                item_column=item_column,
                user_column=str(user_column) if user_column is not None else None,
                on_missing_user=cast(OnMissingUser, on_missing_user),
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
                kind=cast(BoostKind, kind),
                item_column=item_column,
                factor=float(raw.get("factor", 1.0)),
                value_factors=value_factors,
                weight=float(raw.get("weight", 0.0)),
            )
        )
    return boosts


def _parse_blending(raw: dict[str, Any] | None) -> BlendingConfig:
    if not raw:
        return BlendingConfig()
    curve = str(raw.get("curve", "sigmoid"))
    if curve not in BLENDING_CURVES:
        raise ValueError(f"Unknown blending curve {curve!r}; available: {sorted(BLENDING_CURVES)}")
    popular_share = float(raw.get("popular_share", DEFAULT_BLENDING_POPULAR_SHARE))
    if popular_share < 0 or popular_share > 1:
        raise ValueError(f"blending.popular_share must be in [0, 1], got {popular_share}")
    steepness = float(raw.get("steepness", DEFAULT_BLENDING_STEEPNESS))
    if steepness <= 0:
        raise ValueError(f"blending.steepness must be > 0, got {steepness}")
    saturate_at = float(raw.get("saturate_at", DEFAULT_BLENDING_SATURATE_AT))
    if saturate_at <= 0:
        raise ValueError(f"blending.saturate_at must be > 0, got {saturate_at}")
    rrf_k = float(raw.get("rrf_k", DEFAULT_BLENDING_RRF_K))
    if rrf_k <= 0:
        raise ValueError(f"blending.rrf_k must be > 0, got {rrf_k}")
    date_columns_raw = raw.get("latest_date_columns", list(DEFAULT_LATEST_DATE_COLUMNS))
    if isinstance(date_columns_raw, str):
        date_columns: tuple[str, ...] = (date_columns_raw,)
    else:
        date_columns = tuple(str(c) for c in date_columns_raw)
    return BlendingConfig(
        enabled=bool(raw.get("enabled", False)),
        curve=cast(BlendingCurve, curve),
        midpoint=float(raw.get("midpoint", DEFAULT_BLENDING_MIDPOINT)),
        steepness=steepness,
        saturate_at=saturate_at,
        popular_share=popular_share,
        latest_date_columns=date_columns,
        rrf_k=rrf_k,
    )


def load_feature_config(path: Path | str | None = None) -> FeatureConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    def _columns(key: str) -> list[FeatureColumn]:
        columns: list[FeatureColumn] = []
        for raw_column in raw.get(key, []):
            column = str(raw_column["column"])
            column_type = str(raw_column.get("type", "categorical"))
            if column_type not in FEATURE_COLUMN_TYPES:
                raise ValueError(
                    f"Unknown feature type {column_type!r} for {key} column {column!r}; "
                    f"available: {sorted(FEATURE_COLUMN_TYPES)}"
                )
            columns.append(FeatureColumn(column=column, type=cast(FeatureColumnType, column_type)))
        return columns

    return FeatureConfig(
        event_weights={k: float(v) for k, v in raw.get("event_weights", {}).items()},
        quantity_scaled_events=set(raw.get("quantity_scaled_events", [])),
        event_caps={k: int(v) for k, v in raw.get("event_caps", {}).items()},
        user_features=_columns("user_features"),
        item_features=_columns("item_features"),
        item_availability_filters=list(raw.get("item_availability_filters", [])),
        eligibility=_parse_eligibility(list(raw.get("eligibility", []))),
        boosts=_parse_boosts(_raw_boost_rules(raw)),
        boost_overfetch_factor=_parse_boost_overfetch_factor(raw.get("boost_overfetch_factor")),
        blending=_parse_blending(raw.get("blending")),
    )


def _raw_boost_rules(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept ``[[boost]]`` (canonical) or ``[[boosts]]`` (alias)."""
    if "boost" in raw and "boosts" in raw:
        raise ValueError("features.toml must not define both [[boost]] and [[boosts]]; use [[boost]]")
    if "boosts" in raw:
        return list(raw.get("boosts", []))
    return list(raw.get("boost", []))
