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
from sqlalchemy.types import String

# Created on first seed; a non-empty database without it is not ours to replace.
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


def main() -> None:
    url = make_url(DATABASE_URL)
    # The tables are replaced, so only a dedicated test database may be seeded.
    if not (url.database or "").endswith("_robot"):
        raise SystemExit(f"Refusing to seed {url.database!r}: the database name must end with _robot")
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
        if tables and SEED_MARKER_TABLE not in tables:
            raise SystemExit(
                f"Refusing to seed {url.database!r}: it has tables but no {SEED_MARKER_TABLE} marker, "
                "so it is not a database this script created"
            )
        with engine.begin() as conn:
            conn.execute(text(f'CREATE TABLE IF NOT EXISTS "{SEED_MARKER_TABLE}" (seeded_at TIMESTAMPTZ)'))
        EVENTS.to_sql("events", engine, if_exists="replace", index=False)
        USERS.to_sql(
            "users",
            engine,
            if_exists="replace",
            index=False,
            dtype={"favorite_styles": ARRAY(String)},
        )
        ITEMS.to_sql("items", engine, if_exists="replace", index=False)
    finally:
        engine.dispose()
    target = url.render_as_string()
    print(f"Seeded {len(EVENTS)} events, {len(USERS)} users, {len(ITEMS)} items into {target}")


if __name__ == "__main__":
    main()
