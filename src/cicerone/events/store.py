"""Load existing recommendation rows for incremental merge."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Collection

import pandas as pd
from sqlalchemy import Engine, bindparam, create_engine, text
from sqlalchemy.exc import ProgrammingError

from cicerone.config import IOSettings
from cicerone.io.db_store import DEFAULT_RECOMMENDATIONS_TABLE, MISSING_TABLE_ERRORS
from cicerone.io.options import is_s3_not_found, read_parquet, require_option
from cicerone.io.recommendation_schema import RECOMMENDATION_COLUMNS, USER_COLUMN, recommendations_sql_names

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


def _load_dataset_recommendations(output: IOSettings) -> pd.DataFrame:
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


def _db_table_and_columns(output: IOSettings) -> tuple[str, str, str]:
    return recommendations_sql_names(output.options, default_table=DEFAULT_RECOMMENDATIONS_TABLE)


def _load_db_recommendations(output: IOSettings, *, user_ids: Collection[str] | None = None) -> pd.DataFrame:
    table, columns, user_col = _db_table_and_columns(output)
    engine = _engine_for(require_option(output.options, "database_url", "db"))
    try:
        if user_ids is None:
            frame = pd.read_sql_query(text(f"SELECT {columns} FROM {table}"), engine)
        else:
            ids = sorted({str(user_id) for user_id in user_ids})
            if not ids:
                return empty_recommendations_frame()
            stmt = text(f"SELECT {columns} FROM {table} WHERE {user_col} IN :user_ids").bindparams(
                bindparam("user_ids", expanding=True)
            )
            frame = pd.read_sql_query(stmt, engine, params={"user_ids": ids})
    except MISSING_TABLE_ERRORS as exc:
        message = str(getattr(exc, "orig", exc)).lower()
        if "no such column" in message or ("column" in message and "does not exist" in message):
            logger.warning("Recommendations schema mismatch; treating as empty: %s", exc)
            return empty_recommendations_frame()
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


def load_recommendations_frame(output: IOSettings) -> pd.DataFrame:
    if output.kind == "dataset":
        return _load_dataset_recommendations(output)

    if output.kind == "db":
        return _load_db_recommendations(output)

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")


def load_recommendations_for_users(output: IOSettings, user_ids: Collection[str]) -> pd.DataFrame:
    ids = sorted({str(user_id) for user_id in user_ids})
    if not ids:
        return empty_recommendations_frame()

    if output.kind == "dataset":
        frame = _load_dataset_recommendations(output)
        if frame.empty:
            return frame
        return frame.loc[frame[USER_COLUMN].astype(str).isin(ids)].reset_index(drop=True)

    if output.kind == "db":
        return _load_db_recommendations(output, user_ids=ids)

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")


def count_recommendation_users(output: IOSettings) -> int:
    if output.kind == "dataset":
        try:
            # Project only user_id — avoids materializing the full recommendations frame.
            frame = read_parquet(output.options, "recommendations.parquet", columns=[USER_COLUMN])
        except FileNotFoundError:
            return 0
        except Exception as exc:
            if is_s3_not_found(exc):
                return 0
            message = str(exc).lower()
            if USER_COLUMN in message or "fieldref" in message:
                logger.warning(
                    "Recommendations schema mismatch while counting users; treating as empty: %s", exc
                )
                return 0
            raise
        if frame.empty:
            return 0
        return int(frame[USER_COLUMN].astype(str).nunique())

    if output.kind == "db":
        table, _columns, user_col = _db_table_and_columns(output)
        engine = _engine_for(require_option(output.options, "database_url", "db"))
        try:
            with engine.connect() as conn:
                value = conn.execute(text(f"SELECT COUNT(DISTINCT {user_col}) FROM {table}")).scalar()
        except MISSING_TABLE_ERRORS as exc:
            message = str(getattr(exc, "orig", exc)).lower()
            if isinstance(exc, ProgrammingError) or "does not exist" in message or "no such table" in message:
                return 0
            raise
        return int(value or 0)

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")
