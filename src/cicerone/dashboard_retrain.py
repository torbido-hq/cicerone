"""Ask the existing retrain webhook to start a job. The dashboard holds the token."""

from __future__ import annotations

import http.client
import ipaddress
import logging
import urllib.error
import urllib.request

from cicerone.config import Settings
from cicerone.config.validation import is_loopback_host

logger = logging.getLogger(__name__)

_UNREACHABLE = "Retrain trigger is unreachable"
_BUSY = "A run is already in progress or was just triggered"
_DISABLED = "Retrain trigger is not enabled"
_TIMED_OUT = "Retrain trigger did not answer in time; it may still start, check the latest run"
_DROPPED = "Retrain trigger dropped the connection; it may have started, check the latest run"
_NEEDS_URL = "Set [dashboard].trigger_url to reach a trigger that is not on this host"


def _loopback_for_wildcard(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if not address.is_unspecified:
        return host
    # ::1 always reaches an IPv6 wildcard bind; 127.0.0.1 needs a dual-stack socket.
    return "::1" if address.version == 6 else "127.0.0.1"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


# A followed redirect would resend the bearer token to whatever host it names; a 3xx is a failure.
_OPENER = urllib.request.build_opener(_NoRedirect)


class RetrainError(Exception):
    """The webhook refused the request or could not be reached."""


def retrain_configured(settings: Settings) -> bool:
    """True when the dashboard knows how to reach a retrain webhook."""
    return bool(settings.dashboard.trigger_url) or settings.trigger.enabled


def trigger_base_url(settings: Settings) -> str:
    if settings.dashboard.trigger_url:
        return settings.dashboard.trigger_url
    host = _loopback_for_wildcard(settings.trigger.host)
    # Only a same-host trigger is derived: the token must not cross a network over plain HTTP by default.
    if not is_loopback_host(host):
        logger.warning("Retrain trigger host %r is not loopback and [dashboard].trigger_url is unset", host)
        raise RetrainError(_NEEDS_URL)
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{settings.trigger.port}"


def request_retrain(settings: Settings) -> str:
    """POST `/trigger/retrain`. Returns the operator message."""
    if not retrain_configured(settings):
        raise RetrainError(_DISABLED)
    url = f"{trigger_base_url(settings)}/trigger/retrain"
    headers = {"Content-Length": "0"}
    token = settings.dashboard.trigger_auth_token or settings.trigger.auth_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=b"", method="POST", headers=headers)
    try:
        with _OPENER.open(request, timeout=5) as response:
            if response.status != 202:
                logger.warning("Retrain trigger at %s answered %s", url, response.status)
                raise RetrainError("Retrain trigger failed")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            logger.info("Retrain trigger at %s is busy", url)
            raise RetrainError(_BUSY) from exc
        logger.warning("Retrain trigger at %s answered %s", url, exc.code)
        raise RetrainError("Retrain trigger failed") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            logger.warning("Retrain trigger at %s timed out", url)
            raise RetrainError(_TIMED_OUT) from exc
        logger.warning("Retrain trigger at %s is unreachable: %s", url, exc.reason)
        raise RetrainError(_UNREACHABLE) from exc
    except TimeoutError as exc:
        # The request was sent; the trigger may still accept it after we stop waiting.
        logger.warning("Retrain trigger at %s timed out", url)
        raise RetrainError(_TIMED_OUT) from exc
    except (http.client.HTTPException, OSError) as exc:
        # urllib wraps send errors in URLError; one raised unwrapped is past the send, e.g. reading the reply.
        logger.warning("Retrain trigger at %s dropped the connection: %r", url, exc)
        raise RetrainError(_DROPPED) from exc
    return "Retrain started"
