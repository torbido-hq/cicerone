"""Dashboard: a small read-only status page over the run manifest data
job.run() writes on every run, success or failure (see
cicerone.io.manifest_reader), plus a user_id lookup of precomputed
recommendations from the same output store. Runs as its own
process/entrypoint (`python -m cicerone.dashboard`), independent of
[job].mode.

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

import pandas as pd
import uvicorn
from croniter import CroniterError, croniter
from fastapi import Depends, FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cicerone.config import Settings, load_settings
from cicerone.dashboard_users import load_users
from cicerone.http_auth import require_basic_auth
from cicerone.io.base import ManifestReader, RecommendationReader
from cicerone.io.factory import build_manifest_reader, build_recommendation_reader
from cicerone.io.recommendation_schema import ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))
_LOOKUP_K_CAP = 20


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
            "error": run.get("error"),
        }
    return None


def _empty_recommendations_context(
    *,
    user_id: str = "",
    queried: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "items": [],
        "fallback": False,
        "queried": queried,
        "show_category": False,
        "error": error,
    }


def _lookup_k(settings: Settings) -> int:
    return min(settings.top_k, _LOOKUP_K_CAP)


def _recommendations_rows(recs: pd.DataFrame, *, category_column: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in recs.to_dict(orient="records"):
        rank = record.get(RANK_COLUMN)
        score = record.get(SCORE_COLUMN)
        source = record.get(SOURCE_COLUMN)
        item: dict[str, Any] = {
            "rank": None if rank is None or pd.isna(rank) else int(rank),
            "item_id": str(record.get(ITEM_COLUMN, "")),
            "score": None if score is None or pd.isna(score) else float(score),
            "source": "—" if source is None or pd.isna(source) or source == "" else str(source),
        }
        if category_column is not None:
            category = record.get(category_column)
            item["category"] = None if category is None or pd.isna(category) else str(category)
        rows.append(item)
    return rows


def _join_category(recs: pd.DataFrame, items: pd.DataFrame | None, category_column: str) -> pd.DataFrame:
    if (
        items is None
        or items.empty
        or recs.empty
        or ITEM_COLUMN not in recs.columns
        or ITEM_COLUMN not in items.columns
        or category_column not in items.columns
    ):
        return recs
    extra = items[[ITEM_COLUMN, category_column]].drop_duplicates(subset=[ITEM_COLUMN])
    return recs.merge(extra, on=ITEM_COLUMN, how="left")


def _recommendations_context(
    settings: Settings,
    recommendation_reader: RecommendationReader | None,
    user_id: str,
) -> dict[str, Any]:
    user_id = user_id.strip()
    if not user_id:
        return _empty_recommendations_context()
    if recommendation_reader is None:
        return _empty_recommendations_context(
            user_id=user_id,
            queried=True,
            error="Recommendation store is not available.",
        )

    try:
        recommendation_reader.refresh()
    except Exception:
        logger.exception("Failed to refresh recommendation reader for dashboard lookup")

    k = _lookup_k(settings)
    try:
        recs = recommendation_reader.get_recommendations(user_id, k)
        used_fallback = False
        if recs.empty:
            fallback = recommendation_reader.get_cold_start_fallback(k)
            if not fallback.empty:
                recs = fallback
                used_fallback = True
        category_column = settings.serve.category_column
        if category_column in recs.columns:
            joined = recs
        else:
            joined = _join_category(recs, recommendation_reader.get_items(), category_column)
        show_category = category_column in joined.columns
        return {
            "user_id": user_id,
            "items": _recommendations_rows(
                joined, category_column=category_column if show_category else None
            ),
            "fallback": used_fallback,
            "queried": True,
            "show_category": show_category,
            "error": None,
        }
    except Exception as exc:
        logger.exception("Failed to look up recommendations for user_id=%r", user_id)
        return _empty_recommendations_context(user_id=user_id, queried=True, error=str(exc))


def create_app(
    settings: Settings,
    reader: ManifestReader,
    users: dict[str, str],
    recommendation_reader: RecommendationReader | None = None,
) -> FastAPI:
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
            _recommendations_context(settings, recommendation_reader, user_id),
        )

    @app.get("/dashboard", dependencies=[Depends(auth)])
    def dashboard(request: Request):
        context = _status_context()
        context["refresh_interval_seconds"] = settings.dashboard.refresh_interval_seconds
        context.update(_empty_recommendations_context())
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
    rec_reader = build_recommendation_reader(settings.output)
    app = create_app(settings, reader, users, rec_reader)
    uvicorn.run(app, host=settings.dashboard.host, port=settings.dashboard.port)


if __name__ == "__main__":
    main()
