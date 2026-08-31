"""Standalone Basic-Auth status page (`cicerone dashboard`); independent of job.mode."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import uvicorn
from croniter import CroniterError, croniter
from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cicerone.config as config_pkg
from cicerone.config import Settings, load_settings
from cicerone.config.constants import DEFAULT_LOG_FORMAT
from cicerone.dashboard_config import config_display
from cicerone.dashboard_experiments import clear_promotion, experiment_context, promote_winner
from cicerone.dashboard_lookup import lookup_inspector
from cicerone.dashboard_users import load_users
from cicerone.http_auth import require_basic_auth
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


def _is_static_path(path: str) -> bool:
    return path == "/static" or path.startswith("/static/")


def _experiments_redirect(**query: str) -> RedirectResponse:
    target = _EXPERIMENTS_PATH
    if query:
        target = f"{target}?{urlencode(query)}"
    target = target.replace("\\", "")
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc or parsed.path != _EXPERIMENTS_PATH:
        return RedirectResponse(url=_EXPERIMENTS_PATH, status_code=303)
    return RedirectResponse(url=target, status_code=303)


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
    app = FastAPI(title="cicerone-dashboard", docs_url=None, redoc_url=None, openapi_url=None)
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
        return _TEMPLATES.TemplateResponse(request, "dashboard.html", context)

    @app.get("/dashboard/experiments", dependencies=[Depends(auth)])
    def experiments(
        request: Request, message: str = Query(default=""), promote_error: str = Query(default="")
    ):
        context = experiment_context(settings)
        context["message"] = message or None
        context["promote_error"] = promote_error or None
        return _TEMPLATES.TemplateResponse(request, "experiments.html", context)

    @app.post("/dashboard/experiments/promote", dependencies=[Depends(auth)])
    def experiments_promote(variant: str = Form(...)):
        error = promote_winner(settings, variant.strip())
        if error:
            return _experiments_redirect(promote_error=error)
        return _experiments_redirect(message=f"Promoted {variant.strip()}")

    @app.post("/dashboard/experiments/unpromote", dependencies=[Depends(auth)])
    def experiments_unpromote():
        error = clear_promotion(settings)
        if error:
            return _experiments_redirect(promote_error=error)
        return _experiments_redirect(message="Resumed split")

    @app.get("/dashboard/config", dependencies=[Depends(auth)])
    def config_page(request: Request):
        return _TEMPLATES.TemplateResponse(
            request,
            "config.html",
            config_display(
                settings,
                config_path=config_path,
                usernames=tuple(users),
            ),
        )

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
