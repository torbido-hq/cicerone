"""Inspect store tables in tests, on any backend."""

from __future__ import annotations

from sqlalchemy import Engine, Table, inspect
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable


def stored_column_types(engine: Engine, table: str) -> list[tuple[str, str, bool]]:
    return [
        (column["name"], type(column["type"]).__name__.upper(), column["nullable"])
        for column in inspect(engine).get_columns(table)
    ]


def assert_mysql_engine_and_charset(*tables: Table) -> None:
    # Compiled, not read back: a real server's defaults would hide missing table options.
    for table in tables:
        created = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert "ENGINE=InnoDB" in created
        assert "CHARSET=utf8mb4" in created
