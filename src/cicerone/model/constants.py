"""Shared constants for the model package."""

from __future__ import annotations

RANDOM_STATE = 42
DEFAULT_MODELS = ["collaborative", "item_based", "popular"]
LATEST_WINDOW_DAYS = 14
# Reciprocal rank fusion constant (Cormack et al., 2009); default for rrf_k.
RRF_K = 60
SOURCE_COLUMN = "source"
WEIGHT_COLUMN = "_weight"  # internal-only; dropped before returning to callers

COLLABORATIVE_EPOCHS = 30  # LightFMWrapperModel.fit() runs these in one fit_partial
# ProcessPool: LightFM num_threads=1 to avoid workers × BLAS oversubscription.
LIGHTFM_NUM_THREADS_SEQUENTIAL = 4
LIGHTFM_NUM_THREADS_PARALLEL = 1
