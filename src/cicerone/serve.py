"""Serve mode: read API over precomputed recommendations (no live inference)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from cicerone.config import Settings, load_settings
from cicerone.feature_config import FeatureConfig, load_feature_config
from cicerone.http_auth import optional_bearer_deps
from cicerone.io.base import ManifestReader, RecommendationReader
from cicerone.io.factory import build_manifest_reader, build_recommendation_reader
from cicerone.io.recommendation_reader import normalize_items_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _start_refresh_loop(reader: RecommendationReader, interval_seconds: float) -> None:
    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            reader.refresh()

    threading.Thread(target=_loop, daemon=True).start()


def _generated_at(manifest_reader: ManifestReader | None) -> str | None:
    if manifest_reader is None:
        return None
    latest = manifest_reader.read_latest()
    if not latest:
        return None
    value = latest.get("generated_at")
    return str(value) if value is not None else None


def _configure_reader_item_filters(
    reader: RecommendationReader,
    *,
    category_column: str | None,
    availability_filters: Sequence[str],
) -> None:
    configure = getattr(reader, "configure_item_filters", None)
    if callable(configure):
        configure(
            category_column=category_column,
            availability_filters=availability_filters,
        )


def _available_item_ids(items: pd.DataFrame, availability_filters: Sequence[str]) -> frozenset[str] | None:
    if not availability_filters or "item_id" not in items.columns:
        return None
    mask = pd.Series(True, index=items.index)
    for column in availability_filters:
        if column not in items.columns:
            continue
        mask &= items[column]
    return frozenset(items.loc[mask, "item_id"].tolist())


class _ItemsFilterCache:
    """Reuse one normalized items snapshot between refreshes."""

    def __init__(
        self,
        reader: RecommendationReader,
        *,
        category_column: str,
        availability_filters: Sequence[str],
    ) -> None:
        self._reader = reader
        self._category_column = category_column
        self._availability_filters = list(availability_filters)
        self._version: int | None = None
        self._items: pd.DataFrame | None = None
        self._available_ids: frozenset[str] | None = None

    def get(self) -> tuple[pd.DataFrame | None, frozenset[str] | None]:
        version = self._reader.items_version()
        if version == self._version:
            return self._items, self._available_ids

        items = normalize_items_snapshot(
            self._reader.get_items(),
            category_column=self._category_column,
            availability_filters=self._availability_filters,
        )
        available = (
            _available_item_ids(items, self._availability_filters)
            if items is not None and not items.empty
            else None
        )
        self._version = version
        self._items = items
        self._available_ids = available
        return items, available


def _filter_recommendations(
    recs: pd.DataFrame,
    *,
    items: pd.DataFrame | None,
    available_ids: frozenset[str] | None,
    category: str | None,
    category_column: str,
    exclude_unavailable: bool,
    on_missing_category_column: Callable[[], None] | None = None,
) -> pd.DataFrame:
    if recs.empty:
        return recs
    out = recs
    if items is None or items.empty:
        return out.reset_index(drop=True)

    item_ids = out["item_id"].astype(str)

    if category is not None:
        if category_column not in items.columns:
            if on_missing_category_column is not None:
                on_missing_category_column()
            return out.iloc[0:0].reset_index(drop=True)
        allowed = set(items.loc[items[category_column] == str(category), "item_id"])
        out = out[item_ids.isin(allowed)]
        item_ids = out["item_id"].astype(str)

    if exclude_unavailable and available_ids is not None:
        out = out[item_ids.isin(available_ids)]

    return out.reset_index(drop=True)


def create_app(
    settings: Settings,
    reader: RecommendationReader,
    *,
    manifest_reader: ManifestReader | None = None,
    feature_config: FeatureConfig | None = None,
) -> FastAPI:
    app = FastAPI(title="cicerone-serve")
    dependencies = optional_bearer_deps(settings.serve_auth_token)
    availability_filters = list(feature_config.item_availability_filters) if feature_config else []
    category_column = settings.serve_category_column
    # Reader filters are configured by main() before create_app; the cache still
    # normalizes for test doubles / callers that skip that step.
    items_cache = _ItemsFilterCache(
        reader,
        category_column=category_column,
        availability_filters=availability_filters,
    )
    missing_category_warned = False

    def _warn_missing_category_column() -> None:
        nonlocal missing_category_warned
        if missing_category_warned:
            return
        missing_category_warned = True
        logger.warning(
            "Serve category filter requested but items have no column %r — returning empty",
            category_column,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/recommendations/{user_id}", dependencies=dependencies)
    def get_recommendations(
        user_id: str,
        limit: int | None = Query(default=None, gt=0),
        k: int | None = Query(default=None, gt=0, description="Alias for limit (back-compat)"),
        category: str | None = Query(default=None),
        exclude_unavailable: bool = Query(default=True),
    ) -> JSONResponse:
        top_k = limit or k or settings.serve_default_k
        items, available_ids = items_cache.get()
        can_filter = bool(
            items is not None
            and not items.empty
            and (category is not None or (exclude_unavailable and availability_filters))
        )
        fetch_k = max(top_k * 5, top_k) if can_filter else top_k  # over-fetch only if filters apply
        recs = reader.get_recommendations(user_id, fetch_k)
        used_fallback = False
        if recs.empty:
            used_fallback = True
            recs = reader.get_cold_start_fallback(fetch_k)
        if recs.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations for user_id={user_id!r}")

        filtered = _filter_recommendations(
            recs,
            items=items,
            available_ids=available_ids,
            category=category,
            category_column=category_column,
            exclude_unavailable=exclude_unavailable,
            on_missing_category_column=_warn_missing_category_column,
        )
        filtered = filtered.head(top_k).reset_index(drop=True)
        if not filtered.empty:
            filtered = filtered.copy()
            filtered["rank"] = range(1, len(filtered) + 1)

        body: dict[str, Any] = {
            "generated_at": _generated_at(manifest_reader),
            "user_id": user_id,
            "fallback": used_fallback,
            "items": [
                {
                    "item_id": row["item_id"],
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "source": row["source"],
                }
                for _, row in filtered.iterrows()
            ],
        }
        headers = {}
        if body["generated_at"] is not None:
            headers["X-Generated-At"] = str(body["generated_at"])
        return JSONResponse(content=body, headers=headers)

    return app


def main() -> None:
    settings = load_settings()
    if settings.mode != "serve":
        raise RuntimeError(f"job.mode is {settings.mode!r}; python -m cicerone.serve requires mode = 'serve'")

    reader = build_recommendation_reader(settings.output)
    manifest_reader = build_manifest_reader(settings.output)
    try:
        feature_config = load_feature_config(settings.feature_config_path)
    except Exception:
        logger.exception("Failed to load feature config for serve filters; continuing without them")
        feature_config = None

    availability_filters = list(feature_config.item_availability_filters) if feature_config else []
    _configure_reader_item_filters(
        reader,
        category_column=settings.serve_category_column,
        availability_filters=availability_filters,
    )
    _start_refresh_loop(reader, settings.serve_refresh_interval_seconds)

    app = create_app(
        settings,
        reader,
        manifest_reader=manifest_reader,
        feature_config=feature_config,
    )
    uvicorn.run(app, host=settings.serve_host, port=settings.serve_port)


if __name__ == "__main__":
    main()
