"""RecommendationPublisher protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class PublishError(RuntimeError):
    """Sidecar publish, connect, or close failed."""


class RecommendationPublisher(Protocol):
    def connect(self) -> None: ...

    def publish(self, df: pd.DataFrame, *, user_ids: Sequence[str] | None = None) -> None: ...

    def close(self) -> None: ...
