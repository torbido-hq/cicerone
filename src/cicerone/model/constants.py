"""Shared constants for the model package."""

from __future__ import annotations

RANDOM_STATE = 42

# ProcessPool: LightFM num_threads=1 to avoid workers × BLAS oversubscription.
LIGHTFM_NUM_THREADS_SEQUENTIAL = 4  # single-process fit, not the sequential strategy
LIGHTFM_NUM_THREADS_PARALLEL = 1
