"""Package attributes imported from their submodule on first access."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from typing import Any


def lazy_attribute(package: str, sources: Mapping[str, str], name: str) -> Any:
    source = sources.get(name)
    if source is None:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(source), name)
    setattr(sys.modules[package], name, value)
    return value
