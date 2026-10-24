from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from support.fakes import FakeGuard, FakeManifest, FakeReader
from support.system_spec import dashboard_users

from cicerone.config import Settings, load_settings
from cicerone.dashboard import create_app as create_dashboard_app
from cicerone.serve import create_app as create_serve_app
from cicerone.trigger import create_app as create_trigger_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        mode = "serve"

        [job.trigger]
        enabled = true
        auth_token = "trigger-token"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path}"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path}"

        [serve]
        auth_token = "serve-token"
        metrics_enabled = true
        metrics_token = "metrics-token"
        log_impressions = true

        [events]
        enabled = true
        kind = "webhook"

        [track]
        enabled = true

        [dashboard]
        enabled = true
        """
    )
    return load_settings(str(config_path))


def _routes(app: FastAPI) -> list[str]:
    rows = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or {"MOUNT"}
        rows.extend(f"{method} {route.path}" for method in methods)
    return sorted(rows)


def test_serve_routes_with_every_feature_enabled(settings):
    app = create_serve_app(settings, FakeReader(pd.DataFrame()), manifest_reader=FakeManifest())
    assert _routes(app) == [
        "GET /docs",
        "GET /docs/oauth2-redirect",
        "GET /health",
        "GET /metrics",
        "GET /openapi.json",
        "GET /recommendations/{user_id}",
        "GET /redoc",
        "HEAD /docs",
        "HEAD /docs/oauth2-redirect",
        "HEAD /openapi.json",
        "HEAD /redoc",
        "POST /events",
        "POST /track",
    ]


def test_dashboard_routes(settings):
    app = create_dashboard_app(settings, FakeManifest(), dashboard_users())
    assert _routes(app) == [
        "GET /dashboard",
        "GET /dashboard/config",
        "GET /dashboard/experiments",
        "GET /dashboard/logout",
        "GET /dashboard/quality",
        "GET /health",
        "GET /partials/recommendations",
        "GET /partials/status",
        "GET /robots.txt",
        "MOUNT /static",
        "POST /dashboard/experiments/promote",
        "POST /dashboard/experiments/unpromote",
    ]


def test_trigger_routes(settings):
    assert _routes(create_trigger_app(settings, FakeGuard())) == [
        "GET /health",
        "POST /trigger/retrain",
    ]
