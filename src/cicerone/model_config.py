"""RecTools ``model_from_config`` dicts for Cicerone strategies (no ML imports)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Avoid import cycles with config.load_settings.
_DEFAULT_ITEM_BASED_K = 20
_LATEST_WINDOW_DAYS = 14

RECTOOLS_STRATEGY_NAMES: tuple[str, ...] = (
    "collaborative",
    "item_based",
    "popular",
    "latest",
)

DEFAULT_COLLABORATIVE_CONFIG: dict[str, Any] = {
    "cls": "LightFMWrapperModel",
    "epochs": 30,
    "num_threads": 4,
    "model": {
        "no_components": 64,
        "loss": "warp",
        "learning_rate": 0.05,
        "item_alpha": 1e-6,
        "user_alpha": 1e-6,
        "random_state": 42,
    },
}

DEFAULT_ITEM_BASED_CONFIG: dict[str, Any] = {
    "cls": "ImplicitItemKNNWrapperModel",
    "model": {
        "cls": "TFIDFRecommender",
        "K": _DEFAULT_ITEM_BASED_K,
    },
}

DEFAULT_POPULAR_CONFIG: dict[str, Any] = {
    "cls": "PopularModel",
}

DEFAULT_LATEST_CONFIG: dict[str, Any] = {
    "cls": "PopularModel",
    "popularity": "n_interactions",
    "period": {"days": _LATEST_WINDOW_DAYS},
}


def default_model_configs() -> dict[str, dict[str, Any]]:
    """Fresh copy of built-in RecTools configs per strategy."""
    return {
        "collaborative": deepcopy(DEFAULT_COLLABORATIVE_CONFIG),
        "item_based": deepcopy(DEFAULT_ITEM_BASED_CONFIG),
        "popular": deepcopy(DEFAULT_POPULAR_CONFIG),
        "latest": deepcopy(DEFAULT_LATEST_CONFIG),
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into a copy of ``base`` (dicts only)."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _nested_get(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _nested_set(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = config
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value


def item_based_k_from_config(config: dict[str, Any]) -> int | None:
    """``model.K`` from an item_based RecTools config, if present."""
    value = _nested_get(config, "model", "K")
    if value is None:
        return None
    return int(value)


def apply_legacy_item_based_k_neighbors(
    configs: dict[str, dict[str, Any]],
    *,
    k_neighbors: int | None,
    k_neighbors_explicit: bool,
) -> dict[str, dict[str, Any]]:
    """Map legacy ``job.item_based.k_neighbors`` → RecTools ``model.K``."""
    from cicerone.config import ConfigError

    result = {name: deepcopy(cfg) for name, cfg in configs.items()}
    item_cfg = result.setdefault("item_based", deepcopy(DEFAULT_ITEM_BASED_CONFIG))
    native_k = item_based_k_from_config(item_cfg)
    native_was_explicit = bool(item_cfg.pop("_native_k_explicit", False))

    if k_neighbors_explicit and native_was_explicit and native_k is not None:
        if k_neighbors is not None and int(k_neighbors) != int(native_k):
            raise ConfigError(
                f"Conflicting item_based neighbor settings: job.item_based.k_neighbors="
                f"{k_neighbors} vs model.item_based.model.K={native_k}. "
                "Use only one (prefer [model.item_based] RecTools keys)."
            )
    elif k_neighbors is not None and (k_neighbors_explicit or native_k is None):
        _nested_set(item_cfg, ("model", "K"), int(k_neighbors))

    result["item_based"] = item_cfg
    return result


def resolve_model_configs(
    raw_model_section: dict[str, Any] | None = None,
    *,
    legacy_k_neighbors: int | None = None,
    legacy_k_neighbors_explicit: bool = False,
) -> dict[str, dict[str, Any]]:
    """Merge defaults, ``[model.*]`` TOML, and legacy key translations."""
    from cicerone.config import ConfigError

    configs = default_model_configs()
    raw = raw_model_section or {}
    unknown = sorted(name for name in raw if name not in RECTOOLS_STRATEGY_NAMES)
    if unknown:
        raise ConfigError(
            f"[model] contains unknown strategy table(s) {unknown}; "
            f"RecTools-configurable strategies: {list(RECTOOLS_STRATEGY_NAMES)}. "
            "content_fallback is configured under [job.content_fallback], not [model]."
        )

    for name, override in raw.items():
        if not isinstance(override, dict):
            raise ConfigError(f"[model.{name}] must be a table, got {type(override).__name__}")
        override_copy = dict(override)
        if item_based_k_from_config(override_copy) is not None and name == "item_based":
            override_copy["_native_k_explicit"] = True
        configs[name] = deep_merge(configs[name], override_copy)

    configs = apply_legacy_item_based_k_neighbors(
        configs,
        k_neighbors=legacy_k_neighbors if legacy_k_neighbors is not None else _DEFAULT_ITEM_BASED_K,
        k_neighbors_explicit=legacy_k_neighbors_explicit,
    )

    for name, cfg in configs.items():
        if "cls" not in cfg:
            raise ConfigError(f"[model.{name}] is missing required key 'cls'")

    return configs
