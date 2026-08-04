"""Serve mode: read API over precomputed recommendations (no live inference)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import Settings, load_settings
from cicerone.feature_config import FeatureConfig, load_feature_config
from cicerone.http_auth import optional_bearer_deps
from cicerone.io.base import ManifestReader, RecommendationReader
from cicerone.io.factory import build_manifest_reader, build_recommendation_reader

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


def _filter_recommendations(
    recs: pd.DataFrame,
    *,
    items: pd.DataFrame | None,
    category: str | None,
    category_column: str,
    exclude_unavailable: bool,
    availability_filters: list[str],
) -> pd.DataFrame:
    if recs.empty:
        return recs
    out = recs
    if items is None or items.empty:
        return out.reset_index(drop=True)

    item_ids = out["item_id"].astype(str)
    items_by_id = items.copy()
    items_by_id["item_id"] = items_by_id["item_id"].astype(str)

    if category is not None:
        if category_column not in items_by_id.columns:
            logger.warning(
                "Serve category filter requested but items have no column %r — returning empty",
                category_column,
            )
            return out.iloc[0:0].reset_index(drop=True)
        allowed = set(items_by_id.loc[items_by_id[category_column].astype(str) == str(category), "item_id"])
        out = out[item_ids.isin(allowed)]
        item_ids = out["item_id"].astype(str)

    if exclude_unavailable and availability_filters:
        mask = pd.Series(True, index=items_by_id.index)
        for column in availability_filters:
            if column not in items_by_id.columns:
                continue
            mask &= items_by_id[column].fillna(False).astype(bool)
        available = set(items_by_id.loc[mask, "item_id"])
        out = out[item_ids.isin(available)]

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
        # Over-fetch before category/availability filters so limit still fills when possible.
        fetch_k = top_k if (category is None and not exclude_unavailable) else max(top_k * 5, top_k)
        recs = reader.get_recommendations(user_id, fetch_k)
        used_fallback = False
        if recs.empty:
            recs = reader.get_recommendations(COLD_START_USER_ID, fetch_k)
            used_fallback = True
            if recs.empty and hasattr(reader, "get_cold_start_fallback"):
                recs = reader.get_cold_start_fallback(fetch_k)  # type: ignore[attr-defined]
                used_fallback = True
        if recs.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations for user_id={user_id!r}")

        items = reader.get_items() if hasattr(reader, "get_items") else None  # type: ignore[attr-defined]
        filtered = _filter_recommendations(
            recs,
            items=items,
            category=category,
            category_column=category_column,
            exclude_unavailable=exclude_unavailable,
            availability_filters=availability_filters,
        )
        filtered = filtered.head(top_k).reset_index(drop=True)
        # Re-number ranks after filtering so the response stays 1..n.
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
