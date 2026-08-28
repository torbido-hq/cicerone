"""RecTools ``model_from_config`` dicts for Cicerone strategies (no ML imports)."""

from __future__ import annotations

import importlib.util
import logging
from copy import deepcopy
from functools import cache
from typing import Any

from cicerone.config.constants import DEFAULT_ITEM_BASED_K_NEIGHBORS

logger = logging.getLogger(__name__)

# Single source for latest PopularModel period; re-exported via model.constants.
LATEST_WINDOW_DAYS = 14

SEQUENTIAL_STRATEGY = "sequential"
SEQUENTIAL_ARCHITECTURES: dict[str, str] = {
    "sasrec": "SASRecModel",
    "bert4rec": "BERT4RecModel",
    "hstu": "HSTUModel",
}
_SEQUENTIAL_CLS_TO_ARCHITECTURE = {cls: name for name, cls in SEQUENTIAL_ARCHITECTURES.items()}
# Cicerone-only keys; strip before RecTools model_from_config.
_CICERONE_MODEL_KEYS = frozenset({"architecture"})
SEQUENTIAL_EXTRA_HINT = "install with: pip install 'cicerone-recommender[sequential]'"

RECTOOLS_STRATEGY_NAMES: tuple[str, ...] = (
    "collaborative",
    "item_based",
    "sequential",
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
        "K": DEFAULT_ITEM_BASED_K_NEIGHBORS,
    },
}

DEFAULT_POPULAR_CONFIG: dict[str, Any] = {
    "cls": "PopularModel",
}

DEFAULT_LATEST_CONFIG: dict[str, Any] = {
    "cls": "PopularModel",
    "popularity": "n_interactions",
    "period": {"days": LATEST_WINDOW_DAYS},
}

# eSASRec: SASRec objective + LiGR layers + sampled softmax (RecTools 0.17+).
DEFAULT_SEQUENTIAL_CONFIG: dict[str, Any] = {
    "cls": "SASRecModel",
    "architecture": "sasrec",
    "n_factors": 64,
    "n_blocks": 2,
    "n_heads": 4,
    "epochs": 3,
    "loss": "sampled_softmax",
    "n_negatives": 256,
    "transformer_layers_type": "LiGRLayers",
    "session_max_len": 50,
    "train_min_user_interactions": 2,
    "batch_size": 128,
    "lr": 0.001,
    "deterministic": True,
    "verbose": 0,
}


@cache
def sequential_extra_available() -> bool:
    """True when torch + pytorch-lightning are importable (rectools[torch])."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("pytorch_lightning") is not None
    )


_LIGR_LAYERS = "rectools.models.nn.transformers.ligr.LiGRLayers"


def rectools_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy a strategy config with Cicerone-only keys removed."""
    result = deepcopy(config)
    for key in _CICERONE_MODEL_KEYS:
        result.pop(key, None)
    # RecTools import_object needs a dotted path, not the short class name.
    if result.get("transformer_layers_type") == "LiGRLayers":
        result["transformer_layers_type"] = _LIGR_LAYERS
    return result


def apply_sequential_architecture(
    config: dict[str, Any],
    *,
    architecture_explicit: bool = False,
    cls_explicit: bool = False,
    loss_explicit: bool = False,
) -> dict[str, Any]:
    """Map ``architecture`` ↔ RecTools ``cls`` for the sequential strategy."""
    from cicerone.config import ConfigError

    result = deepcopy(config)
    architecture = result.get("architecture")
    cls = result.get("cls")
    if architecture is not None:
        if not isinstance(architecture, str):
            raise ConfigError(
                f"model.sequential.architecture must be a string, got {type(architecture).__name__}"
            )
        key = architecture.lower()
        if key not in SEQUENTIAL_ARCHITECTURES:
            raise ConfigError(
                f"model.sequential.architecture must be one of {list(SEQUENTIAL_ARCHITECTURES)}, "
                f"got {architecture!r}"
            )
        expected_cls = SEQUENTIAL_ARCHITECTURES[key]
        if architecture_explicit and cls_explicit and cls is not None and cls != expected_cls:
            raise ConfigError(
                f"Conflicting sequential settings: architecture={architecture!r} vs cls={cls!r}. "
                'Use only one (prefer architecture = "sasrec", "bert4rec", or "hstu").'
            )
        if architecture_explicit or not cls_explicit:
            result["architecture"] = key
            result["cls"] = expected_cls
            return _constrain_sequential_architecture(result, loss_explicit=loss_explicit)
    if isinstance(cls, str) and cls in _SEQUENTIAL_CLS_TO_ARCHITECTURE:
        result["architecture"] = _SEQUENTIAL_CLS_TO_ARCHITECTURE[cls]
        return _constrain_sequential_architecture(result, loss_explicit=loss_explicit)
    if cls is None:
        result["architecture"] = "sasrec"
        result["cls"] = SEQUENTIAL_ARCHITECTURES["sasrec"]
        return _constrain_sequential_architecture(result, loss_explicit=loss_explicit)
    raise ConfigError(f"model.sequential.cls must be SASRecModel, BERT4RecModel, or HSTUModel, got {cls!r}")


def _constrain_sequential_architecture(
    config: dict[str, Any], *, loss_explicit: bool = False
) -> dict[str, Any]:
    # HSTU uses its own encoder; LiGR layers are SASRec/BERT4Rec-only.
    if config.get("architecture") == "hstu":
        config.pop("transformer_layers_type", None)
        if config.get("loss") == "sampled_softmax" and not loss_explicit:
            logger.warning(
                "HSTU inherited loss=sampled_softmax from SASRec defaults; using softmax "
                "(set [model.sequential].loss explicitly to keep sampled_softmax)"
            )
            config["loss"] = "softmax"
            config.pop("n_negatives", None)
        config.setdefault("relative_time_attention", True)
    return config


def default_model_configs() -> dict[str, dict[str, Any]]:
    """Fresh copy of built-in RecTools configs per strategy."""
    return {
        "collaborative": deepcopy(DEFAULT_COLLABORATIVE_CONFIG),
        "item_based": deepcopy(DEFAULT_ITEM_BASED_CONFIG),
        "sequential": deepcopy(DEFAULT_SEQUENTIAL_CONFIG),
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

    sequential_override: dict[str, Any] = {}
    for name, override in raw.items():
        if not isinstance(override, dict):
            raise ConfigError(f"[model.{name}] must be a table, got {type(override).__name__}")
        override_copy = dict(override)
        if name == SEQUENTIAL_STRATEGY:
            sequential_override = override_copy
        if item_based_k_from_config(override_copy) is not None and name == "item_based":
            override_copy["_native_k_explicit"] = True
        configs[name] = deep_merge(configs[name], override_copy)

    configs = apply_legacy_item_based_k_neighbors(
        configs,
        k_neighbors=legacy_k_neighbors if legacy_k_neighbors is not None else DEFAULT_ITEM_BASED_K_NEIGHBORS,
        k_neighbors_explicit=legacy_k_neighbors_explicit,
    )
    if SEQUENTIAL_STRATEGY in configs:
        configs[SEQUENTIAL_STRATEGY] = apply_sequential_architecture(
            configs[SEQUENTIAL_STRATEGY],
            architecture_explicit="architecture" in sequential_override,
            cls_explicit="cls" in sequential_override,
            loss_explicit="loss" in sequential_override,
        )

    for name, cfg in configs.items():
        if "cls" not in cfg:
            raise ConfigError(f"[model.{name}] is missing required key 'cls'")

    return configs
