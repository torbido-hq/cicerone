#!/usr/bin/env python3
"""Example: ServeClient against a live serve process."""

from __future__ import annotations

import os
import sys

from cicerone.serve_client import ServeClient, ServeClientError


def main() -> int:
    base_url = os.environ.get("CICERONE_SERVE_URL", "http://localhost:8000")
    token = os.environ.get("CICERONE_SERVE_TOKEN")
    user_id = os.environ.get("CICERONE_USER_ID", "alice")

    client = ServeClient(base_url, token=token)
    print("health:", client.health())
    try:
        body = client.recommendations(user_id, limit=5)
    except ServeClientError as exc:
        print(f"recommendations failed: {exc}", file=sys.stderr)
        return 1
    print(f"user={body.user_id} fallback={body.fallback} generated_at={body.generated_at}")
    for row in body.items:
        print(f"  #{row.rank} {row.item_id} score={row.score:.4f} source={row.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
