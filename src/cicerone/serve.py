"""Serve mode: a lightweight read API over precomputed recommendations.

This is deliberately NOT live inference -- there is no model loaded here,
and no lightfm/rectools/implicit import anywhere in this module or its
request path. It only reads whatever the batch job (cicerone.job, run via
cron and/or cicerone.scheduler's trigger) already wrote to the configured
output store, via cicerone.io.recommendation_reader. A serve-only deployment
can therefore run without the training dependencies installed.

Selected via `job.mode = "serve"` in cicerone.toml (see config/cicerone.toml
for the full [serve] section). Run with `python -m cicerone.serve`.
"""

from __future__ import annotations

import logging
import threading
import time

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query

from cicerone.config import Settings, load_settings
from cicerone.http_auth import require_bearer_token
from cicerone.io.base import RecommendationReader
from cicerone.io.factory import build_recommendation_reader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _start_refresh_loop(reader: RecommendationReader, interval_seconds: float) -> None:
    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            reader.refresh()

    threading.Thread(target=_loop, daemon=True).start()


def create_app(settings: Settings, reader: RecommendationReader) -> FastAPI:
    app = FastAPI(title="cicerone-serve")
    auth = require_bearer_token(settings.serve_auth_token) if settings.serve_auth_token else None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    dependencies = [Depends(auth)] if auth else []

    @app.get("/recommendations/{user_id}", dependencies=dependencies)
    def get_recommendations(user_id: str, k: int | None = Query(default=None, gt=0)) -> list[dict]:
        top_k = k or settings.serve_default_k
        recs = reader.get_recommendations(user_id, top_k)
        if recs.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations for user_id={user_id!r}")
        return recs.to_dict(orient="records")

    return app


def main() -> None:
    settings = load_settings()
    if settings.mode != "serve":
        raise RuntimeError(f"job.mode is {settings.mode!r}; python -m cicerone.serve requires mode = 'serve'")

    reader = build_recommendation_reader(settings.output)
    _start_refresh_loop(reader, settings.serve_refresh_interval_seconds)

    app = create_app(settings, reader)
    uvicorn.run(app, host=settings.serve_host, port=settings.serve_port)


if __name__ == "__main__":
    main()
