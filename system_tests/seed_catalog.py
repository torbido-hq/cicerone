"""Seeds a tiny events/users/items catalog into Postgres for the Robot
Framework system tests (docker-compose.robot.yml). Standalone script (no
import of cicerone/tests.support) — runs in the Dockerfile "runtime" stage,
which already has pandas/sqlalchemy/psycopg installed.

User ids below are asserted against directly in system_tests/*.robot —
keep them in sync if you change this data.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, inspect, make_url, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.types import String

# Written with the seed; a non-empty database without a matching one is not ours to replace.
SEED_MARKER_TABLE = "cicerone_robot_seed"

DATABASE_URL = os.environ.get(
    "ROBOT_DATABASE_URL",
    "postgresql+psycopg://cicerone:cicerone@postgres:5432/cicerone_robot",
)

EVENTS = pd.DataFrame(
    [
        {"user_id": "robot-u1", "item_id": "robot-i1", "event_type": "purchase", "quantity": 2},
        {"user_id": "robot-u1", "item_id": "robot-i2", "event_type": "view", "quantity": 1},
        {"user_id": "robot-u2", "item_id": "robot-i1", "event_type": "saved", "quantity": 1},
        {"user_id": "robot-u2", "item_id": "robot-i3", "event_type": "view", "quantity": 1},
        {"user_id": "robot-u3", "item_id": "robot-i2", "event_type": "cart_add", "quantity": 1},
    ]
)
EVENTS["occurred_at"] = pd.Timestamp.now(tz="UTC")

USERS = pd.DataFrame(
    [
        {"user_id": "robot-u1", "favorite_styles": ["ipa"], "region_slug": "lazio"},
        {"user_id": "robot-u2", "favorite_styles": ["lager"], "region_slug": "toscana"},
        {"user_id": "robot-u3", "favorite_styles": [], "region_slug": None},
    ]
)

ITEMS = pd.DataFrame(
    [
        {"item_id": "robot-i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
        {"item_id": "robot-i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
        {"item_id": "robot-i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
    ]
)


def _owned(conn, tables: set[str]) -> bool:
    if not tables:
        return True
    if SEED_MARKER_TABLE not in tables:
        return False
    # The marker records the database OID, so a copied or unrelated database with the table is refused.
    current = conn.execute(text("SELECT oid FROM pg_database WHERE datname = current_database()")).scalar()
    try:
        stored = conn.execute(text(f'SELECT database_oid FROM "{SEED_MARKER_TABLE}"')).scalars().all()
    except ProgrammingError:
        return False
    return stored == [current]


def main() -> None:
    url = make_url(DATABASE_URL)
    # The tables are replaced, so only a dedicated test database may be seeded.
    if not (url.database or "").endswith("_robot"):
        raise SystemExit(f"Refusing to seed {url.database!r}: the database name must end with _robot")
    engine = create_engine(url)
    try:
        # One transaction: a failed seed leaves neither replaced tables nor the marker behind.
        with engine.begin() as conn:
            if not _owned(conn, set(inspect(conn).get_table_names())):
                raise SystemExit(
                    f"Refusing to seed {url.database!r}: it has tables but no {SEED_MARKER_TABLE} marker "
                    "for this database, so it is not one this script created"
                )
            EVENTS.to_sql("events", conn, if_exists="replace", index=False)
            USERS.to_sql(
                "users",
                conn,
                if_exists="replace",
                index=False,
                dtype={"favorite_styles": ARRAY(String)},
            )
            ITEMS.to_sql("items", conn, if_exists="replace", index=False)
            conn.execute(text(f'DROP TABLE IF EXISTS "{SEED_MARKER_TABLE}"'))
            conn.execute(
                text(
                    f'CREATE TABLE "{SEED_MARKER_TABLE}" AS SELECT oid AS database_oid, now() AS seeded_at '
                    "FROM pg_database WHERE datname = current_database()"
                )
            )
    finally:
        engine.dispose()
    target = url.render_as_string()
    print(f"Seeded {len(EVENTS)} events, {len(USERS)} users, {len(ITEMS)} items into {target}")


if __name__ == "__main__":
    main()
