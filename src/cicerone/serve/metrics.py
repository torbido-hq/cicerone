"""Moved to :mod:`cicerone.metrics`; kept as an alias for 0.8 callers."""

from __future__ import annotations

import sys

from cicerone import metrics
from cicerone.metrics import *  # noqa: F403

# Same module object, so module state and monkeypatches stay shared.
sys.modules[__name__] = metrics
