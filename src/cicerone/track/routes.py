"""Serve ``POST /track`` webhook for impressions and clicks."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from cicerone.config import Settings
from cicerone.http_auth import optional_bearer_deps
from cicerone.serve.events_routes import _payloads_from_body, _put_pydantic_schema
from cicerone.serve_schemas import (
    ErrorDetail,
    TrackEvent,
    TrackIngestRequest,
    TrackIngestResponse,
    ValidationErrorDetail,
)
from cicerone.track.normalize import TrackNormalizeError, normalize_track
from cicerone.track.store import TrackStore

logger = logging.getLogger(__name__)

TRACK_PATH = "/track"


def attach_track_ingest_openapi(schema: dict[str, Any]) -> None:
    post = schema.get("paths", {}).get(TRACK_PATH, {}).get("post")
    if not isinstance(post, dict):
        return
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    _put_pydantic_schema(components, TrackEvent)
    _put_pydantic_schema(components, TrackIngestRequest)
    post["requestBody"] = {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "title": "TrackIngestBody",
                    "oneOf": [
                        {"$ref": "#/components/schemas/TrackEvent"},
                        {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/TrackEvent"},
                        },
                        {"$ref": "#/components/schemas/TrackIngestRequest"},
                    ],
                }
            }
        },
    }


def mount_track_routes(
    app: FastAPI, settings: Settings, *, store: TrackStore | None = None
) -> TrackStore | None:
    if not settings.track.enabled:
        return None
    track_store = store if store is not None else TrackStore(settings.output)
    app.state.track_store = track_store
    dependencies = optional_bearer_deps(settings.serve.auth_token)

    @app.post(
        TRACK_PATH,
        response_model=TrackIngestResponse,
        dependencies=dependencies,
        status_code=202,
        tags=["track"],
        summary="Ingest recommendation impressions and clicks",
        responses={
            400: {
                "model": ErrorDetail | ValidationErrorDetail,
                "description": "Invalid track payload",
            },
            401: {"model": ErrorDetail, "description": "Missing or invalid bearer token"},
        },
    )
    async def post_track(request: Request) -> TrackIngestResponse:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        payloads = _payloads_from_body(body)
        try:
            if isinstance(body, dict) and "events" in body:
                TrackIngestRequest.model_validate(body)
            else:
                for payload in payloads:
                    TrackEvent.model_validate(payload)
            rows = [normalize_track(payload).as_row() for payload in payloads]
            accepted = track_store.append_rows(rows)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        except TrackNormalizeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return TrackIngestResponse(
            accepted=accepted,
            event_ids=[str(row["event_id"]) for row in rows],
        )

    return track_store
