"""Serve ``POST /track`` webhook for impressions and clicks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from cicerone.config import Settings
from cicerone.events.errors import RedisError
from cicerone.http_auth import optional_bearer_deps
from cicerone.locks import LockLostError, WriterLockBusyError, build_dataset_writer_lock
from cicerone.serve.events_routes import (
    max_body_bytes,
    payloads_from_body,
    put_pydantic_schema,
    read_limited_json,
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
from cicerone.track.store import TrackStore, open_track_store

logger = logging.getLogger(__name__)

TRACK_PATH = "/track"


def _record_accepted_ingest(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "other")
        counts[kind] = counts.get(kind, 0) + 1
    for kind, count in counts.items():
        record_track_ingest(kind=kind, status="accepted", count=count)


def attach_track_ingest_openapi(schema: dict[str, Any]) -> None:
    post = schema.get("paths", {}).get(TRACK_PATH, {}).get("post")
    if not isinstance(post, dict):
        return
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    put_pydantic_schema(components, TrackEvent)
    put_pydantic_schema(components, TrackIngestRequest)
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
    track_store = (
        store
        if store is not None
        else open_track_store(settings, writer_lock=build_dataset_writer_lock(settings))
    )
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
            503: {
                "model": ErrorDetail,
                "description": "Writer lock busy or lost, or the Redis track log is unavailable",
            },
        },
    )
    async def post_track(request: Request) -> TrackIngestResponse:
        try:
            body = await read_limited_json(request, max_body_bytes(settings))
        except HTTPException:
            record_track_ingest(kind="other", status="error")
            raise
        payloads = payloads_from_body(body)
        try:
            if isinstance(body, dict) and "events" in body:
                TrackIngestRequest.model_validate(body)
            else:
                for payload in payloads:
                    TrackEvent.model_validate(payload)
            rows = [normalize_track(payload).as_row() for payload in payloads]
            accepted_rows = await asyncio.to_thread(track_store.append_accepted_rows, rows)
        except ValidationError as exc:
            record_track_ingest(kind="other", status="error")
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        except TrackNormalizeError as exc:
            record_track_ingest(kind="other", status="error")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except WriterLockBusyError as exc:
            record_track_ingest(kind="other", status="error")
            raise HTTPException(status_code=503, detail="Writer lock is busy") from exc
        except LockLostError as exc:
            record_track_ingest(kind="other", status="error")
            raise HTTPException(status_code=503, detail="Writer lock was lost") from exc
        except RedisError as exc:
            record_track_ingest(kind="other", status="error")
            raise HTTPException(status_code=503, detail="Redis track log is unavailable") from exc
        except Exception:
            record_track_ingest(kind="other", status="error")
            raise
        _record_accepted_ingest(accepted_rows)
        return TrackIngestResponse(
            accepted=len(accepted_rows),
            event_ids=[str(row["event_id"]) for row in accepted_rows],
        )

    return track_store
