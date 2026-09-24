"""RecommendationPublisher protocol."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Protocol

import pandas as pd

from cicerone.io.recommendation_schema import USER_COLUMN

_TOMBSTONE_PUBLISH_ERROR = "publisher.publish must accept user_ids= to emit empty recommendation lists"


class PublishError(RuntimeError):
    """Sidecar publish, connect, or close failed."""


class RecommendationPublisher(Protocol):
    def connect(self) -> None: ...

    def publish(self, df: pd.DataFrame, *, user_ids: Sequence[str] | None = None) -> None: ...

    def close(self) -> None: ...


def _publisher_accepts_user_ids(publish: object) -> bool | None:
    try:
        params = inspect.signature(publish).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return "user_ids" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _tombstone_user_ids(df: pd.DataFrame | None, user_ids: Sequence[str] | None) -> list[str]:
    if not user_ids:
        return []
    present: set[str] = set()
    if df is not None and not df.empty and USER_COLUMN in df.columns:
        present = set(df[USER_COLUMN].astype(str))
    return [str(user_id) for user_id in dict.fromkeys(user_ids) if str(user_id) not in present]


def _publish_without_user_ids(
    publish: Callable[..., None],
    df: pd.DataFrame,
    tombstones: list[str],
    *,
    hide_cause: bool = False,
) -> None:
    if tombstones:
        if df is not None and not df.empty:
            publish(df)
        error = PublishError(_TOMBSTONE_PUBLISH_ERROR)
        if hide_cause:
            raise error from None
        raise error
    publish(df)


def publish_recommendations(
    publisher: RecommendationPublisher,
    df: pd.DataFrame,
    *,
    user_ids: Sequence[str] | None = None,
) -> None:
    """Call ``publish``, passing ``user_ids`` only when the implementation accepts it.

    Empty-list tombstones need ``user_ids``. A one-argument publisher still
    receives any non-empty frame, then this raises ``PublishError`` so mixed
    flushes do not drop live lists.
    """
    publish = publisher.publish
    accepts = _publisher_accepts_user_ids(publish)
    tombstones = _tombstone_user_ids(df, user_ids)
    if accepts is True:
        publish(df, user_ids=user_ids)
        return
    if accepts is False:
        _publish_without_user_ids(publish, df, tombstones)
        return
    try:
        publish(df, user_ids=user_ids)
    except TypeError:
        _publish_without_user_ids(publish, df, tombstones, hide_cause=True)
