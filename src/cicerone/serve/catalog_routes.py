"""User / item / event CRUD against the input catalog store."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cicerone.config import Settings
from cicerone.events.consumed import ConsumedOverlay
from cicerone.http_auth import optional_bearer_deps
from cicerone.io.catalog import CatalogStore, jsonable_row, require_id
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN
from cicerone.serve_schemas import (
    CatalogEventsResponse,
    CatalogRowResponse,
    CatalogWriteResponse,
    ErrorDetail,
    InteractionEvent,
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
    events: list[InteractionEvent] = Field(min_length=1)


def _row_payload(path_id: str, key: str, body: BaseModel) -> dict[str, Any]:
    payload = body.model_dump(exclude_none=True)
    payload[key] = path_id
    return payload


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
                detail="Catalog CRUD requires [input] kind dataset or db",
            )
        return catalog

    @app.put(
        USERS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Upsert a user",
        responses={501: {"model": ErrorDetail}},
    )
    def put_user(user_id: str, body: CatalogUserBody) -> CatalogWriteResponse:
        store = _require()
        store.upsert_user(_row_payload(user_id, USER_COLUMN, body))
        return CatalogWriteResponse(accepted=1)

    @app.get(
        USERS_PATH,
        response_model=CatalogRowResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Get a user",
        responses={404: {"model": ErrorDetail}, 501: {"model": ErrorDetail}},
    )
    def get_user(user_id: str) -> CatalogRowResponse:
        row = _require().get_user(user_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No user {user_id!r}")
        return CatalogRowResponse(row=row)

    @app.delete(
        USERS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Delete a user and their events",
        responses={501: {"model": ErrorDetail}},
    )
    def delete_user(user_id: str) -> CatalogWriteResponse:
        return CatalogWriteResponse(accepted=_require().delete_user(user_id))

    @app.put(
        ITEMS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Upsert an item",
        responses={501: {"model": ErrorDetail}},
    )
    def put_item(item_id: str, body: CatalogItemBody) -> CatalogWriteResponse:
        store = _require()
        store.upsert_item(_row_payload(item_id, ITEM_COLUMN, body))
        return CatalogWriteResponse(accepted=1)

    @app.get(
        ITEMS_PATH,
        response_model=CatalogRowResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Get an item",
        responses={404: {"model": ErrorDetail}, 501: {"model": ErrorDetail}},
    )
    def get_item(item_id: str) -> CatalogRowResponse:
        row = _require().get_item(item_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"No item {item_id!r}")
        return CatalogRowResponse(row=row)

    @app.delete(
        ITEMS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Delete an item",
        responses={501: {"model": ErrorDetail}},
    )
    def delete_item(item_id: str) -> CatalogWriteResponse:
        return CatalogWriteResponse(accepted=_require().delete_item(item_id))

    @app.post(
        CATALOG_EVENTS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Upsert interaction events into the input catalog",
        responses={501: {"model": ErrorDetail}},
    )
    def post_catalog_events(body: CatalogEventsBody) -> CatalogWriteResponse:
        store = _require()
        rows = [event.model_dump() for event in body.events]
        try:
            for row in rows:
                require_id(row, USER_COLUMN)
                require_id(row, ITEM_COLUMN)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        accepted = store.upsert_events(rows)
        if overlay is not None:
            overlay.add_many([(str(row[USER_COLUMN]), str(row[ITEM_COLUMN])) for row in rows])
        return CatalogWriteResponse(accepted=accepted)

    @app.get(
        CATALOG_USER_EVENTS_PATH,
        response_model=CatalogEventsResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="List a user's recent catalog events",
        responses={501: {"model": ErrorDetail}},
    )
    def get_catalog_events(
        user_id: str,
        limit: int = Query(default=50, gt=0, le=1000),
    ) -> CatalogEventsResponse:
        frame = _require().get_events_for_user(user_id, limit)
        events = [jsonable_row(row) for row in frame.to_dict(orient="records")] if not frame.empty else []
        return CatalogEventsResponse(user_id=user_id, events=events)

    @app.delete(
        CATALOG_USER_EVENTS_PATH,
        response_model=CatalogWriteResponse,
        dependencies=dependencies,
        tags=["catalog"],
        summary="Delete a user's events (optionally one item)",
        responses={501: {"model": ErrorDetail}},
    )
    def delete_catalog_events(
        user_id: str,
        item_id: str | None = Query(default=None),
    ) -> CatalogWriteResponse:
        return CatalogWriteResponse(accepted=_require().delete_events_for_user(user_id, item_id=item_id))
