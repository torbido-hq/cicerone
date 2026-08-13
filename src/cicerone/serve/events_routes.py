"""Serve ``POST /events`` webhook route (incremental ingest)."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from cicerone.config import Settings
from cicerone.events.normalize import EventNormalizeError
from cicerone.events.webhook import WebhookEventSource
from cicerone.http_auth import optional_bearer_deps
from cicerone.serve_schemas import ErrorDetail, EventsIngestRequest, EventsIngestResponse

EVENTS_PATH = "/events"


def mount_events_routes(
    app: FastAPI,
    settings: Settings,
    *,
    event_source: WebhookEventSource | None = None,
) -> WebhookEventSource | None:
    """Register webhook ingest when ``[events]`` is enabled with ``kind = "webhook"``."""
    if not (settings.events.enabled and settings.events.kind == "webhook"):
        return None

    webhook_source = event_source or WebhookEventSource(settings.events.options)
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
        },
    )
    async def post_events(request: Request) -> EventsIngestResponse:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        if isinstance(body, list):
            payloads = body
        elif isinstance(body, dict) and isinstance(body.get("events"), list):
            payloads = body["events"]
        elif isinstance(body, dict):
            payloads = [body]
        else:
            raise HTTPException(
                status_code=400,
                detail='Body must be an event object, a list of events, or {"events": [...]}',
            )
        try:
            accepted = webhook_source.ingest(payloads)
        except EventNormalizeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return EventsIngestResponse(
            accepted=len(accepted),
            event_ids=[event.event_id for event in accepted],
        )

    # Keep batch request model in the OpenAPI schema components.
    _ = EventsIngestRequest
    return webhook_source
