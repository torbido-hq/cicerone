"""Shared bearer-token auth dependency for cicerone's HTTP surfaces (the
serve mode's read API, the retrain trigger webhook). A single shared secret
per surface, configured in TOML via "${ENV_VAR}" like every other secret in
this repo. There's no user/session concept and no rate-limiting here (see
docs/architecture.md) -- if that's ever needed, put a reverse proxy in
front rather than growing this module.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=True)


def require_bearer_token(expected_token: str):
    """Returns a FastAPI dependency that rejects requests unless their
    "Authorization: Bearer <token>" header matches `expected_token`."""

    def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),  # noqa: B008
    ) -> None:
        if not hmac.compare_digest(credentials.credentials, expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return _dependency
