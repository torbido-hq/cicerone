"""Ask the existing retrain webhook to start a job. The dashboard holds the token."""

from __future__ import annotations

import urllib.error
import urllib.request

from cicerone.config import Settings

_UNREACHABLE = "Retrain trigger is unreachable"
_BUSY = "A run is already in progress or was just triggered"
_DISABLED = "Retrain trigger is not enabled"


class RetrainError(Exception):
    """The webhook refused the request or could not be reached."""


def trigger_base_url(settings: Settings) -> str:
    host = settings.trigger.host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{settings.trigger.port}"


def request_retrain(settings: Settings) -> str:
    """POST `/trigger/retrain`. Returns the operator message."""
    if not settings.trigger.enabled:
        raise RetrainError(_DISABLED)
    url = f"{trigger_base_url(settings)}/trigger/retrain"
    headers = {"Content-Length": "0"}
    token = settings.trigger.auth_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=b"", method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 202:
                raise RetrainError("Retrain trigger failed")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RetrainError(_BUSY) from exc
        raise RetrainError("Retrain trigger failed") from exc
    except urllib.error.URLError as exc:
        raise RetrainError(_UNREACHABLE) from exc
    return "Retrain started"
