"""Write the serve API OpenAPI document (FastAPI-generated) to stdout or a file.

Usage::

    cicerone export-openapi
    cicerone export-openapi -o docs/openapi/serve.openapi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from cicerone.config import EventsSettings, make_settings
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.recommendation_reader import RECOMMENDATION_COLUMNS
from cicerone.serve import create_app


class _SchemaReader(BaseRecommendationReader):
    """Empty recommendation reader used only to instantiate the FastAPI app for schema export."""

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        del user_id, k, variant
        return pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS))


def build_openapi() -> dict:
    # Bearer token so the exported schema documents HTTP Bearer auth.
    # Enable webhook events so POST /events appears in the checked-in OpenAPI.
    settings = make_settings(
        mode="serve",
        serve_auth_token="openapi-export",
        events=EventsSettings(enabled=True, kind="webhook"),
    )
    return create_app(settings, _SchemaReader()).openapi()


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
