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


def require_incremental_publisher(
    publisher: RecommendationPublisher | None,
) -> RecommendationPublisher | None:
    if publisher is None:
        return None
    publish = getattr(publisher, "publish", None)
    if not callable(publish) or _publisher_accepts_user_ids(publish) is False:
        raise TypeError("IncrementalUpdater publisher.publish must accept user_ids=")
    return publisher


def _publisher_accepts_user_ids(publish: object) -> bool | None:
    try:
        signature = inspect.signature(publish)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if signature is None:
        return None
    params = signature.parameters
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
) -> None:
    if tombstones:
        if df is not None and not df.empty:
            publish(df)
        raise PublishError(_TOMBSTONE_PUBLISH_ERROR)
    publish(df)


def publish_recommendations(
    publisher: RecommendationPublisher,
    df: pd.DataFrame,
    *,
    user_ids: Sequence[str] | None = None,
) -> None:
    """Call ``publish``, passing ``user_ids`` unless the signature is legacy.

    Empty-list tombstones need ``user_ids``. A one-argument publisher whose
    signature is inspectable as legacy still receives any non-empty frame, then
    this raises ``PublishError`` so mixed flushes do not drop live lists.
    When the signature cannot be inspected, ``user_ids`` is passed once; a
    ``TypeError`` from that call is not retried as a one-argument fallback.
    That ``TypeError`` is raised as ``PublishError`` so incremental write
    treats it as a non-retryable sidecar failure.
    """
    publish = publisher.publish
    accepts = _publisher_accepts_user_ids(publish)
    if accepts is False:
        _publish_without_user_ids(publish, df, _tombstone_user_ids(df, user_ids))
        return
    try:
        publish(df, user_ids=user_ids)
    except TypeError as exc:
        raise PublishError(str(exc) or "publisher.publish failed") from exc
