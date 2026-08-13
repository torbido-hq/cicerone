"""Serve ``POST /events`` webhook route (incremental ingest)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from cicerone.config import Settings
from cicerone.events.base import EventBackpressureError
from cicerone.events.normalize import EventNormalizeError
from cicerone.events.webhook import WebhookEventSource
from cicerone.http_auth import optional_bearer_deps
from cicerone.serve_schemas import (
    ErrorDetail,
    EventsIngestRequest,
    EventsIngestResponse,
    InteractionEvent,
)

logger = logging.getLogger(__name__)

EVENTS_PATH = "/events"


def _payloads_from_body(body: object) -> list[object]:
    if isinstance(body, list):
        return list(body)
    if isinstance(body, dict) and isinstance(body.get("events"), list):
        return list(body["events"])
    if isinstance(body, dict):
        return [body]
    raise HTTPException(
        status_code=400,
        detail='Body must be an event object, a list of events, or {"events": [...]}',
    )


def mount_events_routes(
    app: FastAPI,
    settings: Settings,
    *,
    event_source: WebhookEventSource | None = None,
) -> WebhookEventSource | None:
    """Register webhook ingest when ``[events]`` is enabled with ``kind = "webhook"``."""
    if not (settings.events.enabled and settings.events.kind == "webhook"):
        return None

    if event_source is None:
        # OpenAPI export / route unit tests; production passes the bootstrap source.
        logger.warning(
            "Mounting webhook events without an injected EventSource; "
            "pass event_source from start_events_runtime in serve main()"
        )
        webhook_source = WebhookEventSource(settings.events.options)
    else:
        webhook_source = event_source
    app.state.event_source = webhook_source
    events_token = settings.events.options.get("auth_token") or settings.serve.auth_token
    events_dependencies = optional_bearer_deps(str(events_token) if events_token else None)

    @app.post(
        EVENTS_PATH,
        response_model=EventsIngestResponse,
        dependencies=events_dependencies,
        status_code=202,
        tags=["events"],
        summary="Ingest interaction events for incremental updates",
        responses={
            400: {"model": ErrorDetail, "description": "Invalid event payload"},
            401: {"model": ErrorDetail, "description": "Missing or invalid bearer token"},
            429: {"model": ErrorDetail, "description": "Event backlog full"},
        },
    )
    async def post_events(request: Request) -> EventsIngestResponse:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        payloads = _payloads_from_body(body)
        try:
            # Validate shape with the OpenAPI models (single or batch).
            if isinstance(body, dict) and "events" in body:
                EventsIngestRequest.model_validate(body)
            else:
                for payload in payloads:
                    InteractionEvent.model_validate(payload)
            accepted = webhook_source.ingest(payloads)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        except EventBackpressureError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except EventNormalizeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return EventsIngestResponse(
            accepted=len(accepted),
            event_ids=[event.event_id for event in accepted],
        )

    return webhook_source
