"""Shared auth dependencies for cicerone's HTTP surfaces: a single shared
bearer token per surface (serve mode's read API, the retrain trigger
webhook) via require_bearer_token, or HTTP Basic Auth against a small,
fixed set of named users (the dashboard, see cicerone.dashboard_users) via
require_basic_auth. There's no session/cookie concept in either case and no
rate-limiting here (see docs/architecture.md) -- if that's ever needed, put
a reverse proxy in front rather than growing this module.
"""

from __future__ import annotations

import hmac

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=True)
_basic_scheme = HTTPBasic(auto_error=True)

# A fixed, never-matching bcrypt hash checked whenever the supplied username
# isn't in `users`, so an unknown username takes the same time to reject as
# a wrong password for a known one -- avoids a timing side-channel that
# would otherwise let a caller enumerate valid usernames.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt())


def require_bearer_token(expected_token: str):
    """Returns a FastAPI dependency that rejects requests unless their
    "Authorization: Bearer <token>" header matches `expected_token`."""

    def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),  # noqa: B008
    ) -> None:
        if not hmac.compare_digest(credentials.credentials, expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return _dependency


def require_basic_auth(users: dict[str, str]):
    """Returns a FastAPI dependency that rejects requests unless their HTTP
    Basic Auth credentials match a username/bcrypt-hash pair in `users`
    (see cicerone.dashboard_users). Meant for a small, fixed set of named
    people logging in via a browser -- not for machine-to-machine calls,
    which use require_bearer_token instead.
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
