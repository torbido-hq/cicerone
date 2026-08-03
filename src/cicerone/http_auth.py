"""Shared FastAPI auth: bearer token or HTTP Basic Auth."""

from __future__ import annotations

import hmac

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=True)
_basic_scheme = HTTPBasic(auto_error=True)

# Timing-safe dummy hash for unknown usernames (avoids enumeration).
_DUMMY_HASH = b"$2b$12$vJ2512T3h3Og/ZQ2oX0DOumJjo4aEqRgGJPxpAW4Jv76RPsaH7JUm"


def require_bearer_token(expected_token: str):
    def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),  # noqa: B008
    ) -> None:
        if not hmac.compare_digest(credentials.credentials, expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return _dependency


def optional_bearer_deps(token: str | None) -> list:
    if not token:
        return []
    return [Depends(require_bearer_token(token))]


def require_basic_auth(users: dict[str, str]):
    def _dependency(
        credentials: HTTPBasicCredentials = Depends(_basic_scheme),  # noqa: B008
    ) -> None:
        password_hash = users.get(credentials.username)
        candidate_hash = password_hash.encode("utf-8") if password_hash is not None else _DUMMY_HASH
        password_ok = bcrypt.checkpw(credentials.password.encode("utf-8"), candidate_hash)
        if password_hash is None or not password_ok:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Basic"},
            )

    return _dependency
