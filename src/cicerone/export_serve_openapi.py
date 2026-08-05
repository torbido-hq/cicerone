"""Write the serve API OpenAPI document (FastAPI-generated) to stdout or a file.

Usage::

    PYTHONPATH=src python -m cicerone.export_serve_openapi
    PYTHONPATH=src python -m cicerone.export_serve_openapi -o docs/openapi/serve.openapi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from cicerone.config import IOSettings, Settings
from cicerone.serve import create_app


class _SchemaReader:
    """Empty recommendation reader used only to instantiate the FastAPI app for schema export."""

    def refresh(self) -> None:
        return None

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        del user_id, k
        return pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source"])

    def get_items(self) -> pd.DataFrame | None:
        return None

    def items_version(self) -> int:
        return 0

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        del k
        return pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source"])


def _schema_settings() -> Settings:
    # Include a bearer token so the exported schema documents HTTP Bearer auth.
    return Settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/in"}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
        feature_config_path="/app/config/features.toml",
        top_k=10,
        half_life_days=90,
        cron_schedule="0 3 * * *",
        models=None,
        model_weights=None,
        rrf_k=None,
        save_model_artifact=False,
        max_workers=1,
        epoch_metrics=None,
        automl_enabled=False,
        automl_n_splits=2,
        automl_test_days=14,
        automl_primary_metric="MAP",
        automl_candidates=None,
        mode="serve",
        serve_host="0.0.0.0",
        serve_port=8000,
        serve_auth_token="openapi-export",
        serve_default_k=10,
        serve_refresh_interval_seconds=60,
        serve_category_column="category",
        trigger_enabled=False,
        trigger_host="0.0.0.0",
        trigger_port=8080,
        trigger_auth_token=None,
        trigger_debounce_seconds=60,
        trigger_poll_input_bucket=False,
        trigger_poll_interval_seconds=300,
        dashboard_enabled=False,
        dashboard_host="0.0.0.0",
        dashboard_port=8090,
        dashboard_users_path="/tmp/dashboard_users.toml",
        dashboard_refresh_interval_seconds=30,
        dashboard_history_limit=20,
    )


def build_openapi() -> dict:
    app = create_app(_schema_settings(), _SchemaReader())
    return app.openapi()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the Cicerone serve OpenAPI schema")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path (default: stdout)",
    )
    args = parser.parse_args(argv)
    document = build_openapi()
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
