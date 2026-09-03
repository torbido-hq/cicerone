"""Shared HTTP hardening: security headers, CSRF, constant-time tokens."""

from __future__ import annotations

import hmac
import secrets
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

CSRF_COOKIE = "cicerone_csrf"
CSRF_FORM_FIELD = "csrf_token"

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "frame-ancestors 'none'",
}


def token_equals(provided: str | None, expected: str) -> bool:
    if provided is None:
        return False
    try:
        return hmac.compare_digest(provided, expected)
    except (TypeError, ValueError):
        return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response


def csrf_token_for(request: Request) -> str:
    existing = request.cookies.get(CSRF_COOKIE)
    return existing if existing else secrets.token_urlsafe(32)


def set_csrf_cookie(request: Request, response: Response, token: str) -> None:
    if request.cookies.get(CSRF_COOKIE) == token:
        return
    response.set_cookie(
        CSRF_COOKIE,
        token,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )


def require_csrf(request: Request, form_token: str) -> None:
    cookie = request.cookies.get(CSRF_COOKIE, "")
    if not cookie or not form_token or not token_equals(form_token, cookie):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF check failed")
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    expected = request.url.netloc
    if origin and urlparse(origin).netloc != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF check failed")
    if not origin and referer and urlparse(referer).netloc != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF check failed")
