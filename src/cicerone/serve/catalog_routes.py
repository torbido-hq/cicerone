"""User / item / event CRUD against the input catalog store."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, ValidationError

from cicerone.config import Settings
from cicerone.events.consumed import ConsumedOverlay
from cicerone.events.normalize import EventNormalizeError
from cicerone.http_auth import optional_bearer_deps
from cicerone.io.catalog import CatalogStore, jsonable_row, normalize_event_row
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN
from cicerone.serve.events_routes import (
    _max_body_bytes,
    _put_pydantic_schema,
    _read_limited_json,
)
from cicerone.serve_schemas import (
    CatalogEventsResponse,
    CatalogRowResponse,
    CatalogWriteResponse,
    ErrorDetail,
    InteractionEvent,
    ValidationErrorDetail,
)

logger = logging.getLogger(__name__)

USERS_PATH = "/users/{user_id}"
ITEMS_PATH = "/items/{item_id}"
CATALOG_EVENTS_PATH = "/catalog/events"
CATALOG_USER_EVENTS_PATH = "/catalog/events/{user_id}"


class CatalogUserBody(BaseModel):
    labels: dict[str, object] | list[object] | None = None
    comment: str | None = None

    model_config = {"extra": "allow"}


class CatalogItemBody(BaseModel):
    is_hidden: bool | None = None
    categories: list[str] | None = None
    labels: dict[str, object] | list[object] | None = None
    comment: str | None = None

    model_config = {"extra": "allow"}


class CatalogEventsBody(BaseModel):
    events: list[InteractionEvent] = Field(min_length=1, max_length=1000)


def _row_payload(path_id: str, key: str, body: BaseModel) -> dict[str, Any]:
    payload = body.model_dump(exclude_none=True)
    payload[key] = path_id
    return payload


def _path_id(value: str, key: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail=f"{key} is required")
    return stripped


_CATALOG_WRITE: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorDetail},
    401: {"model": ErrorDetail},
    501: {"model": ErrorDetail},
}
_CATALOG_READ: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorDetail},
    401: {"model": ErrorDetail},
    404: {"model": ErrorDetail},
    501: {"model": ErrorDetail},
}
_CATALOG_EVENTS: dict[int | str, dict[str, Any]] = {
    **_CATALOG_WRITE,
    413: {"model": ErrorDetail},
    422: {"model": ValidationErrorDetail},
}


def attach_catalog_events_openapi(schema: dict[str, Any]) -> None:
    post = schema.get("paths", {}).get(CATALOG_EVENTS_PATH, {}).get("post")
    if not isinstance(post, dict):
        return
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    _put_pydantic_schema(components, InteractionEvent)
    _put_pydantic_schema(components, CatalogEventsBody)
    post["requestBody"] = {
        "required": True,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/CatalogEventsBody"},
            }
        },
    }


def mount_catalog_routes(
    app: FastAPI,
    settings: Settings,
    *,
    catalog: CatalogStore | None,
    overlay: ConsumedOverlay | None = None,
) -> None:
    dependencies = optional_bearer_deps(settings.serve.auth_token)

    def _require() -> CatalogStore:
        if catalog is None:
            raise HTTPException(
                status_code=501,
                detail="Catalog CRUD requires a writable dataset or table-backed db input",
            )
        return catalog

    def _forget_consumed(user_id: str, item_id: str | None = None) -> None:
        if overlay is None:
            return
        overlay.discard(user_id, item_id)

    @app.put(
        USERS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Upsert a user",
        responses=_CATALOG_WRITE,
    )
    def put_user(user_id: str, body: CatalogUserBody) -> CatalogWriteResponse:
        store = _require()
        try:
            store.upsert_user(_row_payload(_path_id(user_id, USER_COLUMN), USER_COLUMN, body))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CatalogWriteResponse(accepted=1)

    @app.get(
        USERS_PATH,
        response_model=CatalogRowResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Get a user",
        responses=_CATALOG_READ,
    )
    def get_user(user_id: str) -> CatalogRowResponse:
        row = _require().get_user(_path_id(user_id, USER_COLUMN))
        if row is None:
            raise HTTPException(status_code=404, detail=f"No user {user_id!r}")
        return CatalogRowResponse(row=row)

    @app.delete(
        USERS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Delete a user and their events",
        responses=_CATALOG_WRITE,
    )
    def delete_user(user_id: str) -> CatalogWriteResponse:
        user_id = _path_id(user_id, USER_COLUMN)
        accepted = _require().delete_user(user_id)
        _forget_consumed(user_id)
        return CatalogWriteResponse(accepted=accepted)

    @app.put(
        ITEMS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Upsert an item",
        responses=_CATALOG_WRITE,
    )
    def put_item(item_id: str, body: CatalogItemBody) -> CatalogWriteResponse:
        store = _require()
        try:
            store.upsert_item(_row_payload(_path_id(item_id, ITEM_COLUMN), ITEM_COLUMN, body))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CatalogWriteResponse(accepted=1)

    @app.get(
        ITEMS_PATH,
        response_model=CatalogRowResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Get an item",
        responses=_CATALOG_READ,
    )
    def get_item(item_id: str) -> CatalogRowResponse:
        row = _require().get_item(_path_id(item_id, ITEM_COLUMN))
        if row is None:
            raise HTTPException(status_code=404, detail=f"No item {item_id!r}")
        return CatalogRowResponse(row=row)

    @app.delete(
        ITEMS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Delete an item",
        responses=_CATALOG_WRITE,
    )
    def delete_item(item_id: str) -> CatalogWriteResponse:
        return CatalogWriteResponse(accepted=_require().delete_item(_path_id(item_id, ITEM_COLUMN)))

    @app.post(
        CATALOG_EVENTS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Upsert interaction events into the input catalog",
        responses=_CATALOG_EVENTS,
    )
    async def post_catalog_events(request: Request) -> CatalogWriteResponse:
        store = _require()
        raw = await _read_limited_json(request, _max_body_bytes(settings))
        try:
            body = CatalogEventsBody.model_validate(raw)
            rows = [normalize_event_row(event.model_dump()) for event in body.events]
            accepted, discard, add = store.replace_events(rows)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except (ValueError, EventNormalizeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if overlay is not None:
            overlay.replace_pairs(discard, add)
        return CatalogWriteResponse(accepted=accepted)

    @app.get(
        CATALOG_USER_EVENTS_PATH,
        response_model=CatalogEventsResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="List a user's recent catalog events",
        responses=_CATALOG_WRITE,
    )
    def get_catalog_events(
        user_id: str,
        limit: int = Query(default=50, gt=0, le=1000),
    ) -> CatalogEventsResponse:
        user_id = _path_id(user_id, USER_COLUMN)
        frame = _require().get_events_for_user(user_id, limit)
        events = [jsonable_row(row) for row in frame.to_dict(orient="records")] if not frame.empty else []
        return CatalogEventsResponse(user_id=user_id, events=events)

    @app.delete(
        CATALOG_USER_EVENTS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Delete a user's events (optionally one item)",
        responses=_CATALOG_WRITE,
    )
    def delete_catalog_events(
        user_id: str,
        item_id: str | None = Query(default=None),
    ) -> CatalogWriteResponse:
        user_id = _path_id(user_id, USER_COLUMN)
        if item_id is not None:
            item_id = _path_id(item_id, ITEM_COLUMN)
        accepted = _require().delete_events_for_user(user_id, item_id=item_id)
        _forget_consumed(user_id, item_id)
        return CatalogWriteResponse(accepted=accepted)
