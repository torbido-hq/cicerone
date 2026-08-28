"""Thin HTTP client for the serve read API (stdlib urllib + shared Pydantic models)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from cicerone.serve_schemas import HealthResponse, RecommendationsResponse, TrackIngestResponse


class ServeClientError(Exception):
    """Raised when the serve API returns a non-success HTTP status."""

    def __init__(self, status_code: int, detail: str, *, body: Any = None) -> None:
        self.status_code = status_code
        self.detail = detail
        self.body = body
        super().__init__(f"HTTP {status_code}: {detail}")


class ServeClient:
    """HTTP client for the Cicerone recommendation API (``cicerone serve``)."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self._request("GET", "/health"))

    def recommendations(
        self,
        user_id: str,
        *,
        limit: int | None = None,
        k: int | None = None,
        category: str | None = None,
        exclude_unavailable: bool | None = None,
    ) -> RecommendationsResponse:
        params: dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if k is not None:
            params["k"] = str(k)
        if category is not None:
            params["category"] = category
        if exclude_unavailable is not None:
            params["exclude_unavailable"] = "true" if exclude_unavailable else "false"
        path = f"/recommendations/{urllib.parse.quote(str(user_id), safe='')}"
        return RecommendationsResponse.model_validate(self._request("GET", path, params=params))

    def track(self, payload: dict[str, Any] | list[dict[str, Any]]) -> TrackIngestResponse:
        return TrackIngestResponse.model_validate(self._request("POST", "/track", json_body=payload))

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            body: Any
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = raw
            detail: Any = None
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("message")
            if not detail:
                detail = raw or exc.reason
            raise ServeClientError(exc.code, str(detail), body=body) from exc
