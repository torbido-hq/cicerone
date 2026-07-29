"""Shared auth dependencies for cicerone's HTTP surfaces: a single shared
bearer token per surface via require_bearer_token, or HTTP Basic Auth
against a small, fixed set of named users via require_basic_auth.
"""

from __future__ import annotations

import hmac

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=True)
_basic_scheme = HTTPBasic(auto_error=True)

# Never-matching hash checked when the username isn't in `users`, so an
# unknown username takes the same time to reject as a wrong password for a
# known one (avoids a username-enumeration timing side-channel). Hardcoded
# rather than computed at import time so startup doesn't pay bcrypt's cost
# and processes/tests share the same literal -- it's never a real credential.
_DUMMY_HASH = b"$2b$12$vJ2512T3h3Og/ZQ2oX0DOumJjo4aEqRgGJPxpAW4Jv76RPsaH7JUm"


def require_bearer_token(expected_token: str):
    """FastAPI dependency rejecting requests unless their
    "Authorization: Bearer <token>" header matches `expected_token`."""

    def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),  # noqa: B008
    ) -> None:
        if not hmac.compare_digest(credentials.credentials, expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return _dependency


def require_basic_auth(users: dict[str, str]):
    """FastAPI dependency rejecting requests unless their HTTP Basic Auth
    credentials match a username/bcrypt-hash pair in `users`.
    """

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
