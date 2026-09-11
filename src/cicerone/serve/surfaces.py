"""Popular / latest / similar / session recommend routes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from cicerone.config import Settings
from cicerone.config.constants import DEFAULT_SERVE_MAX_K
from cicerone.http_auth import optional_bearer_deps
from cicerone.io.recommendation_schema import ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN
from cicerone.io.surfaces_reader import SurfacesReader, similar_as_surface
from cicerone.serve.consumed import consumed_item_ids, drop_consumed
from cicerone.serve.item_filters import filter_recommendations
from cicerone.serve_schemas import (
    ErrorDetail,
    SessionRecommendRequest,
    SessionRecommendResponse,
    SimilarResponse,
    SurfaceItem,
    SurfaceResponse,
)

logger = logging.getLogger(__name__)

POPULAR_PATH = "/popular"
LATEST_PATH = "/latest"
SIMILAR_PATH = "/similar/{item_id}"
SESSION_PATH = "/session/recommendations"


def _top_k(limit: int | None, default_k: int) -> int:
    return min(limit or default_k, DEFAULT_SERVE_MAX_K)


def frame_to_items(frame: pd.DataFrame, source_default: str) -> list[SurfaceItem]:
    if frame.empty:
        return []
    items: list[SurfaceItem] = []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        item_id = str(getattr(row, ITEM_COLUMN, ""))
        if not item_id:
            continue
        rank = getattr(row, RANK_COLUMN, index)
        score = getattr(row, SCORE_COLUMN, 0.0)
        source = getattr(row, SOURCE_COLUMN, source_default)
        items.append(
            SurfaceItem(
                item_id=item_id,
                rank=int(rank) if rank is not None else index,
                score=float(score) if score is not None else 0.0,
                source=str(source or source_default),
            )
        )
    for index, item in enumerate(items, start=1):
        item.rank = index
    return items


def mount_surface_routes(
    app: FastAPI,
    settings: Settings,
    *,
    surfaces: SurfacesReader,
    filter_ctx: Callable[[], dict[str, Any]],
) -> None:
    dependencies = optional_bearer_deps(settings.serve.auth_token)

    def _filter_surface(
        frame: pd.DataFrame, *, category: str | None, exclude_unavailable: bool
    ) -> pd.DataFrame:
        ctx = filter_ctx()
        return filter_recommendations(
            frame,
            items=ctx["items"],
            available_ids=ctx["available_ids"],
            category=category,
            category_column=ctx["category_column"],
            exclude_unavailable=exclude_unavailable,
            ids_by_category=ctx["ids_by_category"],
            on_missing_category_column=ctx["on_missing_category_column"],
        )

    def _maybe_hide(frame: pd.DataFrame, user_id: str | None) -> pd.DataFrame:
        if not user_id or not settings.serve.exclude_consumed:
            return frame
        ctx = filter_ctx()
        consumed = consumed_item_ids(
            user_id,
            history=ctx.get("history"),
            overlay=ctx.get("overlay"),
            lookback=settings.serve.consumed_lookback,
        )
        return drop_consumed(frame, consumed)

    def _surface(frame: pd.DataFrame, source: str, generated_at: str | None) -> SurfaceResponse:
        return SurfaceResponse(generated_at=generated_at, items=frame_to_items(frame, source))

    @app.get(
        POPULAR_PATH,
        response_model=SurfaceResponse,
        dependencies=dependencies,
        tags=["recommendations"],
        summary="Precomputed popular items",
        responses={404: {"model": ErrorDetail, "description": "No popular snapshot"}},
    )
    def get_popular(
        limit: int | None = Query(default=None, gt=0, le=DEFAULT_SERVE_MAX_K),
        category: str | None = Query(default=None),
        exclude_unavailable: bool = Query(default=True),
        user_id: str | None = Query(default=None),
    ) -> SurfaceResponse:
        top_k = _top_k(limit, settings.serve.default_k)
        rows = surfaces.get_popular(max(top_k * 5, top_k))
        rows = _filter_surface(rows, category=category, exclude_unavailable=exclude_unavailable)
        rows = _maybe_hide(rows, user_id).head(top_k)
        if rows.empty:
            raise HTTPException(status_code=404, detail="No popular recommendations")
        return _surface(rows, "popular_fallback", filter_ctx()["generated_at"]())

    @app.get(
        LATEST_PATH,
        response_model=SurfaceResponse,
        dependencies=dependencies,
        tags=["recommendations"],
        summary="Precomputed latest items",
        responses={404: {"model": ErrorDetail, "description": "No latest snapshot"}},
    )
    def get_latest(
        limit: int | None = Query(default=None, gt=0, le=DEFAULT_SERVE_MAX_K),
        category: str | None = Query(default=None),
        exclude_unavailable: bool = Query(default=True),
        user_id: str | None = Query(default=None),
    ) -> SurfaceResponse:
        top_k = _top_k(limit, settings.serve.default_k)
        rows = surfaces.get_latest(max(top_k * 5, top_k))
        rows = _filter_surface(rows, category=category, exclude_unavailable=exclude_unavailable)
        rows = _maybe_hide(rows, user_id).head(top_k)
        if rows.empty:
            raise HTTPException(status_code=404, detail="No latest recommendations")
        return _surface(rows, "latest", filter_ctx()["generated_at"]())

    @app.get(
        SIMILAR_PATH,
        response_model=SimilarResponse,
        dependencies=dependencies,
        tags=["recommendations"],
        summary="Items similar to a catalog item",
        responses={404: {"model": ErrorDetail, "description": "No neighbors for this item"}},
    )
    def get_similar(
        item_id: str,
        limit: int | None = Query(default=None, gt=0, le=DEFAULT_SERVE_MAX_K),
        category: str | None = Query(default=None),
        exclude_unavailable: bool = Query(default=True),
        user_id: str | None = Query(default=None),
    ) -> SimilarResponse:
        top_k = _top_k(limit, settings.serve.default_k)
        neighbors = surfaces.get_similar(item_id, max(top_k * 5, top_k))
        rows = similar_as_surface(neighbors)
        rows = _filter_surface(rows, category=category, exclude_unavailable=exclude_unavailable)
        rows = drop_consumed(rows, {str(item_id)})
        rows = _maybe_hide(rows, user_id).head(top_k)
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"No similar items for item_id={item_id!r}")
        return SimilarResponse(
            generated_at=filter_ctx()["generated_at"](),
            item_id=item_id,
            items=frame_to_items(rows, "item_based"),
        )

    @app.post(
        SESSION_PATH,
        response_model=SessionRecommendResponse,
        dependencies=dependencies,
        tags=["recommendations"],
        summary="Recommend from an anonymous session via item neighbors",
        responses={404: {"model": ErrorDetail, "description": "No session neighbors or popular fallback"}},
    )
    def post_session(body: SessionRecommendRequest) -> SessionRecommendResponse:
        session_ids = [str(item_id) for item_id in body.items if str(item_id).strip()]
        session_ids.extend(str(event.item_id) for event in body.events if event.item_id)
        session_ids = list(dict.fromkeys(session_ids))
        if not session_ids:
            raise HTTPException(status_code=400, detail="Session must include at least one item_id")
        top_k = settings.serve.default_k
        parts: list[pd.DataFrame] = []
        for item_id in session_ids:
            neighbors = similar_as_surface(surfaces.get_similar(item_id, top_k))
            if not neighbors.empty:
                parts.append(neighbors)
        used_fallback = False
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            if SCORE_COLUMN in merged.columns:
                merged = merged.sort_values(SCORE_COLUMN, ascending=False, kind="mergesort")
            merged = merged.drop_duplicates(subset=[ITEM_COLUMN], keep="first")
        else:
            used_fallback = True
            merged = surfaces.get_popular(top_k * 5)
        merged = _filter_surface(merged, category=None, exclude_unavailable=True)
        merged = drop_consumed(merged, set(session_ids)).head(top_k)
        if merged.empty:
            raise HTTPException(status_code=404, detail="No session recommendations")
        return SessionRecommendResponse(
            generated_at=filter_ctx()["generated_at"](),
            fallback=used_fallback,
            items=frame_to_items(merged, "item_based" if not used_fallback else "popular_fallback"),
        )
