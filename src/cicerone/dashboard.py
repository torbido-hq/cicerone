"""Dashboard: a small read-only status page over the same run manifest data
job.run() already writes on every run, success or failure (see
cicerone.io.manifest_reader). Always its own process/entrypoint
(`python -m cicerone.dashboard`), independent of [job].mode -- so it's
available for any deployment topology (batch-only, serve, with or without
the retrain trigger) instead of being tied to whichever of those happens to
already expose an HTTP server.

Rendered server-side (FastAPI + Jinja2 + htmx for polling, a small Stimulus
controller for relative timestamps) rather than a JSON API + JS framework:
this is a handful of maintainers checking "did last night's run succeed",
not an app that needs client-side routing/state.

Protected by HTTP Basic Auth against a small, fixed set of named users
(cicerone.dashboard_users, managed via
`python -m cicerone.manage_dashboard_users`) rather than the single shared
bearer token serve.py/trigger.py use: a handful of people logging in via a
browser is a better fit for named per-person credentials than one shared
machine-to-machine secret. A browser caches Basic Auth credentials per
origin after the first successful login, so htmx's periodic polling
requests below are authenticated automatically -- no token/cookie wiring
needed on the client.
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
    """Whether a scheduled run looks overdue: the last recorded run should
    have been followed by another one, per `cron_schedule`, by now.
    Deliberately unaware of the retrain trigger's extra webhook/s3-poll
    runs -- those only ever happen *in addition to* the cron schedule (see
    trigger.py), so the cron-derived expectation is always a valid lower
    bound on "how fresh should this be".

    `is_stale` is `None` (unknown, not stale/fresh) rather than raising when
    `cron_schedule` is misconfigured -- a bad cron expression shouldn't take
    the whole dashboard view down; `error` then carries a human-readable
    reason the template can surface instead.
    """
    if manifest is None or not manifest.get("generated_at"):
        return {"is_stale": True, "expected_next_run": None, "error": None}

    last_run = datetime.fromisoformat(manifest["generated_at"])
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
        history = reader.read_recent(settings.dashboard_history_limit)
        staleness = _compute_staleness(manifest, settings.cron_schedule, datetime.now(UTC))
        return {"manifest": manifest, "history": history, "staleness": staleness}

    @app.get("/partials/status", dependencies=[Depends(auth)])
    def status_partial(request: Request):
        # `request` is passed positionally first, not via a "request" key in
        # the context dict -- that's the correct call signature for the
        # pinned starlette==1.3.1 (Jinja2Templates.TemplateResponse(self,
        # request, name, context=None, ...), verified via
        # inspect.signature(Jinja2Templates.TemplateResponse)). The older
        # TemplateResponse(name, {"request": request, ...}) form some
        # docs/linters still expect doesn't apply to this pinned version.
        return _TEMPLATES.TemplateResponse(request, "_status.html", _status_context())

    @app.get("/dashboard", dependencies=[Depends(auth)])
    def dashboard(request: Request):
        context = _status_context()
        context["refresh_interval_seconds"] = settings.dashboard_refresh_interval_seconds
        # See status_partial() above re: request-positional-first call form.
        return _TEMPLATES.TemplateResponse(request, "dashboard.html", context)

    return app


def main() -> None:
    settings = load_settings()
    if not settings.dashboard_enabled:
        raise RuntimeError("dashboard.enabled must be true in the loaded config to run cicerone.dashboard")

    users = load_users(settings.dashboard_users_path)
    if not users:
        raise RuntimeError(
            f"No dashboard users configured at {settings.dashboard_users_path!r} -- "
            "add one with `python -m cicerone.manage_dashboard_users add <username>`"
        )

    reader = build_manifest_reader(settings.output)
    app = create_app(settings, reader, users)
    uvicorn.run(app, host=settings.dashboard_host, port=settings.dashboard_port)


if __name__ == "__main__":
    main()
