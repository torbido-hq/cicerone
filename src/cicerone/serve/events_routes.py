"""Serve ``POST /events`` webhook route (incremental ingest)."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from cicerone.config import Settings
from cicerone.config.constants import DEFAULT_EVENTS_MAX_BODY_BYTES
from cicerone.events.base import EventBackpressureError
from cicerone.events.normalize import EventNormalizeError
from cicerone.events.webhook import WebhookEventSource
from cicerone.http_auth import optional_bearer_deps
from cicerone.serve_schemas import (
    ErrorDetail,
    EventsIngestRequest,
    EventsIngestResponse,
    InteractionEvent,
    ValidationErrorDetail,
)

logger = logging.getLogger(__name__)

EVENTS_PATH = "/events"


def _max_body_bytes(settings: Settings) -> int:
    raw = settings.events.options.get("max_body_bytes", DEFAULT_EVENTS_MAX_BODY_BYTES)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_EVENTS_MAX_BODY_BYTES
    return value if value >= 1 else DEFAULT_EVENTS_MAX_BODY_BYTES


async def _read_limited_json(request: Request, max_bytes: int) -> object:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from None
        if declared > max_bytes:
            raise HTTPException(status_code=413, detail="Request body too large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="Request body too large")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be JSON") from exc


def _rewrite_json_schema_defs(node: Any) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return {"$ref": f"#/components/schemas/{ref.rsplit('/', 1)[-1]}"}
        return {key: _rewrite_json_schema_defs(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_rewrite_json_schema_defs(item) for item in node]
    return node


def _put_pydantic_schema(components: dict[str, Any], model: type[BaseModel]) -> None:
    generated = model.model_json_schema(ref_template="#/components/schemas/{model}")
    for name, subschema in (generated.pop("$defs", None) or {}).items():
        components.setdefault(name, _rewrite_json_schema_defs(subschema))
    components[model.__name__] = _rewrite_json_schema_defs(generated)


def attach_events_ingest_openapi(schema: dict[str, Any]) -> None:
    """Document flexible POST /events JSON bodies (handler parses ``Request``)."""
    post = schema.get("paths", {}).get(EVENTS_PATH, {}).get("post")
    if not isinstance(post, dict):
        return
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    _put_pydantic_schema(components, InteractionEvent)
    _put_pydantic_schema(components, EventsIngestRequest)
    post["requestBody"] = {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "title": "EventsIngestBody",
                    "oneOf": [
                        {"$ref": "#/components/schemas/InteractionEvent"},
                        {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/InteractionEvent"},
                        },
                        {"$ref": "#/components/schemas/EventsIngestRequest"},
                    ],
                }
            }
        },
    }


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
            400: {
                "model": ErrorDetail | ValidationErrorDetail,
                "description": (
                    "Invalid event payload: string ``ErrorDetail`` for malformed JSON / "
                    "normalize errors, or ``ValidationErrorDetail`` (Pydantic ``errors()`` list)"
                ),
            },
            401: {"model": ErrorDetail, "description": "Missing or invalid bearer token"},
            413: {"model": ErrorDetail, "description": "Request body too large"},
            429: {"model": ErrorDetail, "description": "Event backlog full"},
        },
    )
    async def post_events(request: Request) -> EventsIngestResponse:
        body = await _read_limited_json(request, _max_body_bytes(settings))
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
