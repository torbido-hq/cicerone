"""Shared constants for the model package."""

from __future__ import annotations

from cicerone.config.constants import DEFAULT_MODELS, RRF_K
from cicerone.model_config import LATEST_WINDOW_DAYS

RANDOM_STATE = 42
SOURCE_COLUMN = "source"
WEIGHT_COLUMN = "_weight"  # internal-only; dropped before returning to callers

COLLABORATIVE_EPOCHS = 30  # LightFMWrapperModel.fit() runs these in one fit_partial
# ProcessPool: LightFM num_threads=1 to avoid workers × BLAS oversubscription.
LIGHTFM_NUM_THREADS_SEQUENTIAL = 4  # single-process fit, not the sequential strategy
LIGHTFM_NUM_THREADS_PARALLEL = 1

__all__ = [
    "COLLABORATIVE_EPOCHS",
    "DEFAULT_MODELS",
    "LATEST_WINDOW_DAYS",
    "LIGHTFM_NUM_THREADS_PARALLEL",
    "LIGHTFM_NUM_THREADS_SEQUENTIAL",
    "RANDOM_STATE",
    "RRF_K",
    "SOURCE_COLUMN",
    "WEIGHT_COLUMN",
]
