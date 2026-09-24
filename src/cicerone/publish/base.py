"""RecommendationPublisher protocol."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class PublishError(RuntimeError):
    """Sidecar publish, connect, or close failed."""


class RecommendationPublisher(Protocol):
    def connect(self) -> None: ...

    def publish(self, df: pd.DataFrame, *, user_ids: Sequence[str] | None = None) -> None: ...

    def close(self) -> None: ...


def publish_recommendations(
    publisher: RecommendationPublisher,
    df: pd.DataFrame,
    *,
    user_ids: Sequence[str] | None = None,
) -> None:
    """Call ``publish``, passing ``user_ids`` only when the implementation accepts it."""
    publish = publisher.publish
    try:
        params = inspect.signature(publish).parameters
    except (TypeError, ValueError):
        try:
            publish(df, user_ids=user_ids)
            return
        except TypeError:
            publish(df)
            return
    if "user_ids" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        publish(df, user_ids=user_ids)
        return
    publish(df)
