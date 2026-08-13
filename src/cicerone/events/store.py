"""Load existing recommendation rows for incremental merge."""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.config import IOSettings
from cicerone.io.db_store import DEFAULT_RECOMMENDATIONS_TABLE
from cicerone.io.options import is_s3_not_found, read_parquet, require_option, sql_identifier
from cicerone.io.recommendation_reader import RECOMMENDATION_COLUMNS

logger = logging.getLogger(__name__)


def empty_recommendations_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS))


def load_recommendations_frame(output: IOSettings) -> pd.DataFrame:
    if output.kind == "dataset":
        try:
            frame = read_parquet(output.options, "recommendations.parquet")
        except FileNotFoundError:
            return empty_recommendations_frame()
        except Exception as exc:
            if is_s3_not_found(exc):
                return empty_recommendations_frame()
            raise
        if frame.empty:
            return empty_recommendations_frame()
        return frame

    if output.kind == "db":
        table = sql_identifier(
            output.options.get("recommendations_table", DEFAULT_RECOMMENDATIONS_TABLE),
            option="recommendations_table",
        )
        engine = create_engine(require_option(output.options, "database_url", "db"), pool_pre_ping=True)
        try:
            frame = pd.read_sql_query(text(f"SELECT * FROM {table}"), engine)
        except Exception:
            logger.exception("Failed to load recommendations from %r; treating as empty", table)
            return empty_recommendations_frame()
        if frame.empty:
            return empty_recommendations_frame()
        return frame

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")
