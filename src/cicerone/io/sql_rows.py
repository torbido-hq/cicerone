"""SQL row helpers shared by the track and experiment db backends."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd
from sqlalchemy import Connection, Engine, TextClause, text

from cicerone.io.db_errors import SQL_READ_ERRORS, is_missing_table_error
from cicerone.io.engines import insert_sql, quote_identifier
from cicerone.locks.hold import WriteFence

logger = logging.getLogger(__name__)


def jsonish(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):
        return None
    return value


def frame_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    records = frame.to_dict(orient="records")
    return [{str(key): jsonish(value) for key, value in row.items()} for row in records]


def read_sql_or_missing(
    engine: Engine, statement: TextClause, table: str, params: Mapping[str, Any] | None = None
) -> pd.DataFrame | None:
    """``None`` when ``table`` does not exist yet."""
    try:
        return pd.read_sql(statement, engine, params=params)
    except SQL_READ_ERRORS as exc:
        if is_missing_table_error(exc):
            return None
        logger.exception("Failed to read table %r", table)
        raise


def append_frame_fenced(engine: Engine, table: str, frame: pd.DataFrame, fence: WriteFence) -> None:
    with engine.begin() as conn:
        fence.ensure()
        frame.to_sql(table, conn, if_exists="append", index=False)
        fence.ensure()


def replace_rows_fenced(conn: Connection, table: str, params: Mapping[str, Any], fence: WriteFence) -> None:
    fence.ensure()
    conn.execute(text(f"DELETE FROM {quote_identifier(conn, table)}"))
    fence.ensure()
    conn.execute(text(insert_sql(conn, table, list(params))), params)
    fence.ensure()
