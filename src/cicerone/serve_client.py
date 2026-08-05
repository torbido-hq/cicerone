"""Thin HTTP client for the serve read API (stdlib only — no extra deps)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ServeClientError(Exception):
    """Raised when the serve API returns a non-success HTTP status."""

    def __init__(self, status_code: int, detail: str, *, body: Any = None) -> None:
        self.status_code = status_code
        self.detail = detail
        self.body = body
        super().__init__(f"HTTP {status_code}: {detail}")


class ServeClient:
    """Minimal client for `GET /health` and `GET /recommendations/{user_id}`.

    Uses only the Python standard library so integrators can copy this file
    or depend on the installed package without pulling httpx/requests.
    """

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

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def recommendations(
        self,
        user_id: str,
        *,
        limit: int | None = None,
        k: int | None = None,
        category: str | None = None,
        exclude_unavailable: bool | None = None,
    ) -> dict[str, Any]:
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
        return self._request("GET", path, params=params)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, method=method, headers=headers)
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
            detail = body.get("detail") if isinstance(body, dict) else (raw or exc.reason)
            raise ServeClientError(exc.code, str(detail), body=body) from exc
