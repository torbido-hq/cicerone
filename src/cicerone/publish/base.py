"""RecommendationPublisher protocol."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class RecommendationPublisher(Protocol):
    def connect(self) -> None: ...

    def publish(self, df: pd.DataFrame) -> None: ...

    def close(self) -> None: ...
