"""Load existing recommendation rows for incremental merge."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import ProgrammingError

from cicerone.config import IOSettings
from cicerone.io.db_store import DEFAULT_RECOMMENDATIONS_TABLE, MISSING_TABLE_ERRORS
from cicerone.io.options import is_s3_not_found, read_parquet, require_option, sql_identifier
from cicerone.io.recommendation_reader import RECOMMENDATION_COLUMNS

logger = logging.getLogger(__name__)

_MAX_CACHED_ENGINES = 8
_engines: OrderedDict[str, Engine] = OrderedDict()
_engines_lock = threading.Lock()


def empty_recommendations_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS))


def dispose_recommendation_engines() -> None:
    """Dispose cached SQLAlchemy engines (tests / process shutdown)."""
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
    for engine in engines:
        engine.dispose()


def _engine_for(database_url: str) -> Engine:
    with _engines_lock:
        engine = _engines.get(database_url)
        if engine is not None:
            _engines.move_to_end(database_url)
            return engine
        engine = create_engine(database_url, pool_pre_ping=True)
        _engines[database_url] = engine
        while len(_engines) > _MAX_CACHED_ENGINES:
            _url, old = _engines.popitem(last=False)
            old.dispose()
        return engine


def _normalize_recommendation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RECOMMENDATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"recommendations frame missing columns: {missing}")
    return frame.loc[:, list(RECOMMENDATION_COLUMNS)]


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
        try:
            return _normalize_recommendation_columns(frame)
        except ValueError as exc:
            logger.warning("Recommendations schema mismatch; treating as empty: %s", exc)
            return empty_recommendations_frame()

    if output.kind == "db":
        table = sql_identifier(
            output.options.get("recommendations_table", DEFAULT_RECOMMENDATIONS_TABLE),
            option="recommendations_table",
        )
        engine = _engine_for(require_option(output.options, "database_url", "db"))
        try:
            frame = pd.read_sql_query(text(f"SELECT * FROM {table}"), engine)
        except MISSING_TABLE_ERRORS as exc:
            message = str(getattr(exc, "orig", exc)).lower()
            if isinstance(exc, ProgrammingError) or "does not exist" in message or "no such table" in message:
                logger.warning("Recommendations table %r missing; treating as empty", table)
                return empty_recommendations_frame()
            raise
        if frame.empty:
            return empty_recommendations_frame()
        try:
            return _normalize_recommendation_columns(frame)
        except ValueError as exc:
            logger.warning("Recommendations schema mismatch; treating as empty: %s", exc)
            return empty_recommendations_frame()

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")
