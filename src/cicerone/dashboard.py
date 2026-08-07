"""Dashboard: a small read-only status page over the run manifest data
job.run() writes on every run, success or failure (see
cicerone.io.manifest_reader). Runs as its own process/entrypoint
(`python -m cicerone.dashboard`), independent of [job].mode.

Rendered server-side (FastAPI + Jinja2 + htmx for polling, a small Stimulus
controller for relative timestamps) rather than a JSON API + JS framework.

Protected by HTTP Basic Auth against a small, fixed set of named users
(cicerone.dashboard_users, managed via
`python -m cicerone.manage_dashboard_users`).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from croniter import CroniterError, croniter
from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cicerone.config import Settings, load_settings
from cicerone.dashboard_users import load_users
from cicerone.http_auth import require_basic_auth
from cicerone.io.base import ManifestReader
from cicerone.io.factory import build_manifest_reader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))


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


def create_app(settings: Settings, reader: ManifestReader, users: dict[str, str]) -> FastAPI:
    app = FastAPI(title="cicerone-dashboard")
    app.mount("/static", StaticFiles(directory=str(_PACKAGE_DIR / "static")), name="static")
    auth = require_basic_auth(users)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    def _status_context() -> dict[str, Any]:
        manifest = reader.read_latest()
        history = reader.read_recent(settings.dashboard.history_limit)
        staleness = _compute_staleness(manifest, settings.cron_schedule, datetime.now(UTC))
        return {"manifest": manifest, "history": history, "staleness": staleness}

    @app.get("/partials/status", dependencies=[Depends(auth)])
    def status_partial(request: Request):
        return _TEMPLATES.TemplateResponse(request, "_status.html", _status_context())

    @app.get("/dashboard", dependencies=[Depends(auth)])
    def dashboard(request: Request):
        context = _status_context()
        context["refresh_interval_seconds"] = settings.dashboard.refresh_interval_seconds
        return _TEMPLATES.TemplateResponse(request, "dashboard.html", context)

    return app


def main() -> None:
    settings = load_settings()
    if not settings.dashboard.enabled:
        raise RuntimeError("dashboard.enabled must be true in the loaded config to run cicerone.dashboard")

    users = load_users(settings.dashboard.users_path)
    if not users:
        raise RuntimeError(
            f"No dashboard users configured at {settings.dashboard.users_path!r} -- "
            "add one with `python -m cicerone.manage_dashboard_users add <username>`"
        )

    reader = build_manifest_reader(settings.output)
    app = create_app(settings, reader, users)
    uvicorn.run(app, host=settings.dashboard.host, port=settings.dashboard.port)


if __name__ == "__main__":
    main()
