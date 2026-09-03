"""Serve ``POST /track`` webhook for impressions and clicks."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from cicerone.config import Settings
from cicerone.http_auth import optional_bearer_deps
from cicerone.serve.events_routes import (
    _max_body_bytes,
    _payloads_from_body,
    _put_pydantic_schema,
    _read_limited_json,
)
from cicerone.serve.metrics import record_track_ingest
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


def _record_accepted_ingest(rows: list[dict[str, Any]], accepted: int) -> None:
    if accepted <= 0:
        return
    if accepted == len(rows):
        counts: dict[str, int] = {}
        for row in rows:
            kind = str(row.get("kind") or "other")
            counts[kind] = counts.get(kind, 0) + 1
        for kind, count in counts.items():
            record_track_ingest(kind=kind, status="accepted", count=count)
        return
    kinds = {str(row.get("kind") or "other") for row in rows}
    kind = kinds.pop() if len(kinds) == 1 else "other"
    record_track_ingest(kind=kind, status="accepted", count=accepted)


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
            413: {"model": ErrorDetail, "description": "Request body too large"},
        },
    )
    async def post_track(request: Request) -> TrackIngestResponse:
        try:
            body = await _read_limited_json(request, _max_body_bytes(settings))
        except HTTPException:
            record_track_ingest(kind="other", status="error")
            raise
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
            record_track_ingest(kind="other", status="error")
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        except TrackNormalizeError as exc:
            record_track_ingest(kind="other", status="error")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            record_track_ingest(kind="other", status="error")
            raise
        _record_accepted_ingest(rows, accepted)
        return TrackIngestResponse(
            accepted=accepted,
            event_ids=[str(row["event_id"]) for row in rows],
        )

    return track_store
