"""Shared missing-value / list-coercion helpers for policy and content features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Sentinel for "attribute absent" (distinct from None / NaN on a present column).
MISSING = object()


def is_sequence_attr(value: object) -> bool:
    if isinstance(value, (list, tuple, set, pd.Series)):
        return True
    return isinstance(value, np.ndarray) and value.ndim > 0


def is_missing(value: object) -> bool:
    """True for None / NaN / pd.NA / NaT / ``MISSING``; False for containers."""
    if value is None or value is MISSING:
        return True
    if is_sequence_attr(value):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (np.ndarray, pd.Series, list)):
        return False
    return bool(result)


def as_list(value: object) -> list:
    if is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return as_list(value.item())
        values = value.tolist()
    elif isinstance(value, (list, tuple, set, pd.Series)):
        values = list(value) if not isinstance(value, pd.Series) else value.tolist()
    else:
        return [value]
    return [v for v in values if not is_missing(v)]


def str_set(values: Iterable) -> set[str]:
    return {str(v) for v in values if not is_missing(v)}


# ``item_true`` / availability string tokens only; avoid ``astype(bool)`` ("false" → True).
ITEM_TRUE_STRINGS = frozenset({"1", "true", "True", "TRUE", "yes", "Yes", "YES"})
ITEM_FALSE_STRINGS = frozenset({"0", "false", "False", "FALSE", "no", "No", "NO", ""})


def coerce_item_true(value: object) -> bool:
    """Return True only for explicit truthy tokens; unknowns are False."""
    if is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    token = str(value).strip()
    if token in ITEM_TRUE_STRINGS:
        return True
    if token in ITEM_FALSE_STRINGS:
        return False
    return False


def item_true_mask(item_values: pd.Series) -> pd.Series:
    """Boolean mask for availability / ``item_true`` without silent string→True coercion."""
    return item_values.map(coerce_item_true).astype(bool)
