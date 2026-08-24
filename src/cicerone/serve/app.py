"""Serve mode: read API over precomputed recommendations (no live inference)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.openapi.utils import get_openapi
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cicerone import __version__
from cicerone.config import Settings, load_settings
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig, load_feature_config
from cicerone.http_auth import optional_bearer_deps
from cicerone.io.base import ManifestReader, RecommendationReader
from cicerone.io.recommendation_reader import SOURCE_COLUMN
from cicerone.reasons import parse_reasons
from cicerone.serve.bootstrap_events import start_events_runtime
from cicerone.serve.code_samples import HEALTH_PATH, RECOMMENDATIONS_PATH, attach_code_samples
from cicerone.serve.events_routes import attach_events_ingest_openapi, mount_events_routes
from cicerone.serve.item_filters import (
    ItemsFilterCache,
    configure_reader_item_filters,
    filter_recommendations,
)
from cicerone.serve.metrics import (
    METRICS_TOKEN_HEADER,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    record_recommendations_served,
    update_cache_age_gauge,
    update_events_source_health,
)
from cicerone.serve_schemas import ErrorDetail, HealthResponse, RecommendationItem, RecommendationsResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SERVE_API_TITLE = "Cicerone Serve API"
SERVE_API_VERSION = __version__
SERVE_API_DESCRIPTION = f"""
Read-only HTTP API over **precomputed** recommendations written by the batch job.

There is no live inference in the request path: `GET {RECOMMENDATIONS_PATH}`
looks up rows already stored in the configured output (dataset parquet or DB).

When `[events]` is enabled with `kind = "webhook"`, `POST /events` accepts
interaction events for micro-batch incremental updates (write-through to the
same output store). See `docs/incremental-events.md`.

Interactive docs: `/docs` (Swagger UI) and `/redoc` (includes language
code samples via ``x-codeSamples``). Machine-readable schema: `/openapi.json`.
A checked-in copy lives at `docs/openapi/serve.openapi.json` (regenerate with
`cicerone export-openapi`).
""".strip()


def _start_refresh_loop(
    reader: RecommendationReader,
    interval_seconds: float,
    *,
    generated_at_cache: _GeneratedAtCache | None = None,
) -> None:
    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            reader.refresh()
            if generated_at_cache is not None:
                generated_at_cache.refresh()

    threading.Thread(target=_loop, daemon=True).start()


def _read_generated_at(manifest_reader: ManifestReader | None) -> str | None:
    if manifest_reader is None:
        return None
    try:
        latest = manifest_reader.read_latest()
    except Exception:
        logger.exception("Failed to read latest manifest for generated_at; omitting header")
        return None
    if not latest:
        return None
    value = latest.get("generated_at")
    return str(value) if value is not None else None


class _GeneratedAtCache:
    """Cache manifest ``generated_at``; refresh with the recommendations loop."""

    def __init__(self, manifest_reader: ManifestReader | None) -> None:
        self._manifest_reader = manifest_reader
        self._lock = threading.Lock()
        self._value: str | None = None
        self.refresh()

    def refresh(self) -> None:
        value = _read_generated_at(self._manifest_reader)
        with self._lock:
            self._value = value

    def get(self) -> str | None:
        with self._lock:
            return self._value


def _route_endpoint(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None:
        return route.path
    return request.url.path


def create_app(
    settings: Settings,
    reader: RecommendationReader,
    *,
    manifest_reader: ManifestReader | None = None,
    feature_config: FeatureConfig | None = None,
    event_source: WebhookEventSource | None = None,
    events_worker: EventWorker | None = None,
) -> FastAPI:
    app = FastAPI(
        title=SERVE_API_TITLE,
        version=SERVE_API_VERSION,
        description=SERVE_API_DESCRIPTION,
    )
    dependencies = optional_bearer_deps(settings.serve.auth_token)
    availability_filters = list(feature_config.item_availability_filters) if feature_config else []
    category_column = settings.serve.category_column
    items_cache = ItemsFilterCache(
        reader,
        category_column=category_column,
        availability_filters=availability_filters,
    )
    generated_at_cache = _GeneratedAtCache(manifest_reader)
    app.state.generated_at_cache = generated_at_cache
    app.state.events_worker = events_worker
    missing_category_warned = False

    @app.middleware("http")
    async def record_request_metrics(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        endpoint = _route_endpoint(request)
        REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=request.method,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        return response

    def _warn_missing_category_column() -> None:
        nonlocal missing_category_warned
        if missing_category_warned:
            return
        missing_category_warned = True
        logger.warning(
            "Serve category filter requested but items have no column %r — returning empty",
            category_column,
        )

    @app.get(
        HEALTH_PATH,
        response_model=HealthResponse,
        tags=["health"],
        summary="Liveness probe",
    )
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    if settings.serve.metrics_enabled:

        @app.get("/metrics", tags=["metrics"], include_in_schema=False)
        def metrics(request: Request) -> Response:
            metrics_token = settings.serve.metrics_token
            if metrics_token and request.headers.get(METRICS_TOKEN_HEADER) != metrics_token:
                raise HTTPException(status_code=401, detail="Invalid or missing metrics token")
            update_cache_age_gauge()
            # Event source lag/connected are refreshed by the worker loop (not on scrape).
            if request.app.state.events_worker is None:
                update_events_source_health(connected=False, lag=None)
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get(
        RECOMMENDATIONS_PATH,
        response_model=RecommendationsResponse,
        dependencies=dependencies,
        tags=["recommendations"],
        summary="Precomputed top-K recommendations for a user",
        responses={
            400: {"model": ErrorDetail, "description": "Conflicting limit and k"},
            401: {"model": ErrorDetail, "description": "Missing or invalid bearer token"},
            404: {"model": ErrorDetail, "description": "No rows and no cold-start fallback"},
        },
    )
    def get_recommendations(
        user_id: str,
        response: Response,
        limit: int | None = Query(default=None, gt=0, description="Top-K rows to return"),
        k: int | None = Query(default=None, gt=0, description="Alias for limit (back-compat)"),
        category: str | None = Query(
            default=None,
            description="Keep only items whose configured category column matches this value",
        ),
        exclude_unavailable: bool = Query(
            default=True,
            description="Re-apply item_availability_filters against the items snapshot",
        ),
    ) -> RecommendationsResponse:
        if limit is not None and k is not None and limit != k:
            raise HTTPException(
                status_code=400,
                detail="limit and k disagree; pass only one (or the same value)",
            )
        if limit is not None:
            top_k = limit
        elif k is not None:
            top_k = k
        else:
            top_k = settings.serve.default_k
        items, available_ids, ids_by_category = items_cache.get()
        can_filter = bool(
            items is not None
            and not items.empty
            and (category is not None or (exclude_unavailable and availability_filters))
        )
        fetch_k = max(top_k * 5, top_k) if can_filter else top_k
        recs = reader.get_recommendations(user_id, fetch_k)
        used_fallback = False
        if recs.empty:
            used_fallback = True
            recs = reader.get_cold_start_fallback(fetch_k)
        if recs.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations for user_id={user_id!r}")

        filtered = filter_recommendations(
            recs,
            items=items,
            available_ids=available_ids,
            category=category,
            category_column=category_column,
            exclude_unavailable=exclude_unavailable,
            ids_by_category=ids_by_category,
            on_missing_category_column=_warn_missing_category_column,
        )
        filtered = filtered.head(top_k).reset_index(drop=True)
        if not filtered.empty:
            filtered = filtered.copy()
            filtered["rank"] = range(1, len(filtered) + 1)
            if SOURCE_COLUMN in filtered.columns:
                record_recommendations_served(set(filtered[SOURCE_COLUMN].astype(str)))

        generated_at = generated_at_cache.get()
        body = RecommendationsResponse(
            generated_at=generated_at,
            user_id=user_id,
            fallback=used_fallback,
            items=[
                RecommendationItem(
                    item_id=str(row.item_id),
                    rank=int(row.rank),
                    score=float(row.score),
                    source=str(row.source),
                    reasons=parse_reasons(getattr(row, "reasons", None)),
                )
                for row in filtered.itertuples(index=False)
            ],
        )
        if generated_at is not None:
            response.headers["X-Generated-At"] = str(generated_at)
        return body

    mount_events_routes(app, settings, event_source=event_source)

    def custom_openapi() -> dict:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("headers", {})["X-Generated-At"] = {
            "description": "ISO timestamp from the last job-run manifest (mirrors body.generated_at)",
            "schema": {"type": "string", "example": "2026-08-04T03:00:00+00:00"},
        }
        rec_responses = (
            schema.get("paths", {}).get(RECOMMENDATIONS_PATH, {}).get("get", {}).get("responses", {})
        )
        ok = rec_responses.get("200")
        if isinstance(ok, dict):
            ok.setdefault("headers", {})["X-Generated-At"] = {
                "$ref": "#/components/headers/X-Generated-At",
            }
        attach_code_samples(schema)
        attach_events_ingest_openapi(schema)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
    return app


def main() -> None:
    from cicerone.io.factory import build_manifest_reader, build_recommendation_reader

    settings = load_settings()
    if settings.mode != "serve":
        raise RuntimeError(f"job.mode is {settings.mode!r}; cicerone serve requires mode = 'serve'")

    reader = build_recommendation_reader(settings.output)
    manifest_reader = build_manifest_reader(settings.output)
    feature_path = Path(settings.feature_config_path)
    if feature_path.is_file():
        feature_config = load_feature_config(feature_path)
    else:
        logger.warning(
            "feature config missing at %s; serve continuing without features.toml",
            feature_path,
        )
        feature_config = None

    availability_filters = list(feature_config.item_availability_filters) if feature_config else []
    configure_reader_item_filters(
        reader,
        category_column=settings.serve.category_column,
        availability_filters=availability_filters,
    )

    events_runtime = start_events_runtime(settings, feature_config=feature_config, reader=reader)
    app = create_app(
        settings,
        reader,
        manifest_reader=manifest_reader,
        feature_config=feature_config,
        event_source=events_runtime.webhook_source,
        events_worker=events_runtime.worker,
    )
    _start_refresh_loop(
        reader,
        settings.serve.refresh_interval_seconds,
        generated_at_cache=app.state.generated_at_cache,
    )
    try:
        uvicorn.run(app, host=settings.serve.host, port=settings.serve.port)
    finally:
        events_runtime.stop()
