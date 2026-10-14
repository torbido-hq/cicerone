"""Deprecated aliases for renamed module attributes."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping
from typing import Any


def renamed_attribute(module: str, renames: Mapping[str, str], name: str) -> Any:
    """Module ``__getattr__`` body; ``renames`` maps old name -> dotted path of the new attribute."""
    target = renames.get(name)
    if target is None:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    warnings.warn(
        f"{module}.{name} is deprecated and will be removed in 0.9.0; use {target}",
        FutureWarning,
        stacklevel=3,
    )
    target_module, _, attr = target.rpartition(".")
    return getattr(importlib.import_module(target_module), attr)
