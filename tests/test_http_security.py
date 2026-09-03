from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from cicerone.http_security import CSRF_COOKIE, require_csrf, set_csrf_cookie, token_equals
from cicerone.io.options import readonly_select


def test_token_equals_rejects_none_and_type_mismatch():
    assert token_equals(None, "secret") is False
    assert token_equals("secret", "secret") is True
    assert token_equals(b"secret", "secret") is False  # type: ignore[arg-type]


def _request(*, headers: list[tuple[bytes, bytes]] | None = None, cookies: str | None = None) -> Request:
    header_list = list(headers or [])
    if cookies is not None:
        header_list.append((b"cookie", cookies.encode("latin-1")))
    return Request(
        {
            "type": "http",
            "asgi": {"spec_version": "2.3", "version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/dashboard/experiments/promote",
            "raw_path": b"/dashboard/experiments/promote",
            "query_string": b"",
            "headers": header_list,
            "client": ("127.0.0.1", 123),
            "server": ("testserver", 80),
        }
    )


def test_require_csrf_rejects_cross_origin():
    request = _request(
        cookies=f"{CSRF_COOKIE}=tok",
        headers=[(b"origin", b"http://evil.example")],
    )
    with pytest.raises(HTTPException) as exc:
        require_csrf(request, "tok")
    assert exc.value.status_code == 403


def test_require_csrf_rejects_cross_site_referer_without_origin():
    request = _request(
        cookies=f"{CSRF_COOKIE}=tok",
        headers=[(b"referer", b"http://evil.example/phish")],
    )
    with pytest.raises(HTTPException) as exc:
        require_csrf(request, "tok")
    assert exc.value.status_code == 403


def test_set_csrf_cookie_skips_when_already_set():
    request = _request(cookies=f"{CSRF_COOKIE}=tok")
    response = Response()
    set_csrf_cookie(request, response, "tok")
    assert "set-cookie" not in {key.lower() for key in response.headers}


def test_readonly_select_rejects_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        readonly_select(123, option="q")  # type: ignore[arg-type]
