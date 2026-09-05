"""Standalone Basic-Auth status page (`cicerone dashboard`); independent of job.mode."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from croniter import CroniterError, croniter
from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cicerone.config as config_pkg
from cicerone.config import Settings, load_settings
from cicerone.config.constants import DEFAULT_LOG_FORMAT
from cicerone.dashboard_config import config_display
from cicerone.dashboard_experiments import clear_promotion, experiment_context, promote_winner
from cicerone.dashboard_lookup import lookup_inspector
from cicerone.dashboard_quality import quality_context
from cicerone.dashboard_users import load_users
from cicerone.http_auth import require_basic_auth
from cicerone.http_security import (
    CSRF_FORM_FIELD,
    FLASH_COOKIE,
    SecurityHeadersMiddleware,
    clear_flash_cookie,
    csrf_token_for,
    parse_flash_cookie,
    require_csrf,
    set_csrf_cookie,
    set_flash_cookie,
)
from cicerone.io.base import ManifestReader, RecommendationReader, UserHistoryReader
from cicerone.io.factory import build_manifest_reader, build_recommendation_reader, build_user_history_reader

logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
logger = logging.getLogger(__name__)

ROBOTS_TAG = "noindex, nofollow, noarchive, nosnippet, noimageindex"
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))
_TEMPLATES.env.globals["robots_tag"] = ROBOTS_TAG
_EXPERIMENTS_PATH = "/dashboard/experiments"
_LOGOUT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Signed out · Cicerone dashboard</title>
</head>
<body>
  <p>Signed out. Close this tab, or clear the saved password for this site in your browser.</p>
  <p><a href="/dashboard">Sign in again</a></p>
</body>
</html>
"""


def _as_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


_TEMPLATES.env.filters["as_percent"] = _as_percent


def _is_static_path(path: str) -> bool:
    return path == "/static" or path.startswith("/static/")


def _chrome(settings: Settings) -> dict[str, Any]:
    return {
        "nav_quality_available": bool(settings.track.enabled or settings.eval.enabled),
        "nav_experiments_available": bool(settings.experiment.enabled),
    }


def _experiments_redirect(
    request: Request | None = None,
    *,
    message: str = "",
    promote_error: str = "",
) -> RedirectResponse:
    response = RedirectResponse(url=_EXPERIMENTS_PATH, status_code=303)
    if request is not None and (message or promote_error):
        set_flash_cookie(request, response, ok=message or None, error=promote_error or None)
    return response


def _compute_staleness(manifest: dict[str, Any] | None, cron_schedule: str, now: datetime) -> dict[str, Any]:
    """Whether a scheduled run looks overdue per cron_schedule (ignore webhook/poll extras)."""
    if manifest is None or not manifest.get("generated_at"):
        return {"is_stale": True, "expected_next_run": None, "error": None}

    generated_at = manifest["generated_at"]
    if isinstance(generated_at, datetime):
        last_run = generated_at
    else:
        try:
            last_run = datetime.fromisoformat(str(generated_at))
        except ValueError as exc:
            logger.warning(
                "Invalid generated_at %r in manifest, can't compute staleness: %s", generated_at, exc
            )
            return {"is_stale": None, "expected_next_run": None, "error": str(exc)}

    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=UTC)

    try:
        expected_next_run = croniter(cron_schedule, last_run).get_next(datetime)
    except CroniterError as exc:
        logger.warning("Invalid cron_schedule %r, can't compute staleness: %s", cron_schedule, exc)
        return {"is_stale": None, "expected_next_run": None, "error": str(exc)}
    is_stale = now > expected_next_run
    return {"is_stale": is_stale, "expected_next_run": expected_next_run.isoformat(), "error": None}


def page_title(
    *,
    user_id: str = "",
    manifest: dict[str, Any] | None = None,
    staleness: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    if manifest is not None:
        if manifest.get("status") == "failed":
            parts.append("failed")
        elif staleness is not None and staleness.get("is_stale"):
            parts.append("stale")
    cleaned = user_id.strip()
    if cleaned:
        parts.append(cleaned)
    parts.append("Cicerone dashboard")
    return " · ".join(parts)


def _html(request: Request, template: str, context: dict[str, Any], *, settings: Settings | None = None):
    token = csrf_token_for(request)
    extra = _chrome(settings) if settings is not None else {}
    context = {**extra, **context, "csrf_token": token}
    response = _TEMPLATES.TemplateResponse(request, template, context)
    set_csrf_cookie(request, response, token)
    return response


def _incremental_status(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Most recent incremental write-through run from manifest history (newest first)."""
    for run in history:
        if run.get("triggered_by") != "incremental":
            continue
        return {
            "status": run.get("status"),
            "generated_at": run.get("generated_at"),
            "last_incremental_at": run.get("last_incremental_at") or run.get("generated_at"),
            "events": run.get("incremental_events_applied", run.get("n_events")),
            "online_users_refreshed": run.get("online_users_refreshed"),
            "online_fit_partial_epochs": run.get("online_fit_partial_epochs"),
            "error": run.get("error"),
        }
    return None


def create_app(
    settings: Settings,
    reader: ManifestReader,
    users: dict[str, str],
    recommendation_reader: RecommendationReader | None = None,
    history_reader: UserHistoryReader | None = None,
    config_path: str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="cicerone-dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.mount("/static", StaticFiles(directory=str(_PACKAGE_DIR / "static")), name="static")
    auth = require_basic_auth(users)

    @app.middleware("http")
    async def anti_index_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Robots-Tag"] = ROBOTS_TAG
        if not _is_static_path(request.url.path):
            response.headers.setdefault("Cache-Control", "private, no-store")
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/robots.txt")
    def robots() -> PlainTextResponse:
        return PlainTextResponse(ROBOTS_TXT)

    def _status_context() -> dict[str, Any]:
        manifest = reader.read_latest()
        history = reader.read_recent(settings.dashboard.history_limit)
        staleness = _compute_staleness(manifest, settings.cron_schedule, datetime.now(UTC))
        return {
            "manifest": manifest,
            "history": history,
            "staleness": staleness,
            "events_enabled": settings.events.enabled,
            "events_kind": settings.events.kind if settings.events.enabled else None,
            "incremental": _incremental_status(history),
        }

    @app.get("/partials/status", dependencies=[Depends(auth)])
    def status_partial(request: Request):
        return _TEMPLATES.TemplateResponse(request, "_status.html", _status_context())

    @app.get("/partials/recommendations", dependencies=[Depends(auth)])
    def recommendations_partial(request: Request, user_id: str = Query(default="")):
        return _TEMPLATES.TemplateResponse(
            request,
            "_recommendations.html",
            lookup_inspector(settings, recommendation_reader, history_reader, user_id),
        )

    @app.get("/dashboard", dependencies=[Depends(auth)])
    def dashboard(request: Request, user_id: str = Query(default="")):
        context = _status_context()
        context["refresh_interval_seconds"] = settings.dashboard.refresh_interval_seconds
        context.update(lookup_inspector(settings, recommendation_reader, history_reader, user_id))
        context["page_title"] = page_title(
            user_id=str(context.get("user_id") or ""),
            manifest=context["manifest"],
            staleness=context["staleness"],
        )
        return _html(request, "dashboard.html", context, settings=settings)

    @app.get("/dashboard/logout")
    def logout() -> HTMLResponse:
        return HTMLResponse(
            _LOGOUT_HTML,
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
        )

    @app.get("/dashboard/experiments", dependencies=[Depends(auth)])
    def experiments(request: Request):
        context = experiment_context(settings)
        message, promote_error = parse_flash_cookie(request.cookies.get(FLASH_COOKIE))
        context["message"] = message
        context["promote_error"] = promote_error
        context["page_title"] = "Experiments · Cicerone dashboard"
        response = _html(request, "experiments.html", context, settings=settings)
        if request.cookies.get(FLASH_COOKIE) is not None:
            clear_flash_cookie(response)
        return response

    @app.get("/dashboard/quality", dependencies=[Depends(auth)])
    def quality(request: Request):
        context = quality_context(settings)
        context["page_title"] = "Quality · Cicerone dashboard"
        return _html(request, "quality.html", context, settings=settings)

    @app.post("/dashboard/experiments/promote", dependencies=[Depends(auth)])
    def experiments_promote(
        request: Request,
        variant: str = Form(...),
        csrf_token: str = Form("", alias=CSRF_FORM_FIELD),
    ):
        require_csrf(request, csrf_token)
        error = promote_winner(settings, variant.strip())
        if error:
            return _experiments_redirect(request, promote_error=error)
        return _experiments_redirect(request, message=f"Promoted {variant.strip()}")

    @app.post("/dashboard/experiments/unpromote", dependencies=[Depends(auth)])
    def experiments_unpromote(
        request: Request,
        csrf_token: str = Form("", alias=CSRF_FORM_FIELD),
    ):
        require_csrf(request, csrf_token)
        error = clear_promotion(settings)
        if error:
            return _experiments_redirect(request, promote_error=error)
        return _experiments_redirect(request, message="Resumed split")

    @app.get("/dashboard/config", dependencies=[Depends(auth)])
    def config_page(request: Request):
        context = config_display(
            settings,
            config_path=config_path,
            usernames=tuple(users),
        )
        context["page_title"] = "Configuration · Cicerone dashboard"
        return _html(request, "config.html", context, settings=settings)

    return app


def main() -> None:
    settings = load_settings()
    if not settings.dashboard.enabled:
        raise RuntimeError("dashboard.enabled must be true in the loaded config to run cicerone.dashboard")

    users = load_users(settings.dashboard.users_path)
    if not users:
        raise RuntimeError(
            f"No dashboard users configured at {settings.dashboard.users_path!r} -- "
            "add one with `cicerone users add <username>`"
        )

    reader = build_manifest_reader(settings.output)
    try:
        rec_reader = build_recommendation_reader(settings.output)
    except Exception:
        logger.exception("Recommendation store is not available; dashboard lookup will be disabled")
        rec_reader = None
    try:
        history_reader = build_user_history_reader(settings.input)
    except Exception:
        logger.exception("Event store is not available; dashboard history pane will be disabled")
        history_reader = None
    loaded_config_path = os.environ.get("CICERONE_CONFIG_PATH") or config_pkg.DEFAULT_CONFIG_PATH
    app = create_app(
        settings,
        reader,
        users,
        rec_reader,
        history_reader,
        config_path=loaded_config_path,
    )
    uvicorn.run(app, host=settings.dashboard.host, port=settings.dashboard.port)


if __name__ == "__main__":
    main()
