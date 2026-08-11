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
