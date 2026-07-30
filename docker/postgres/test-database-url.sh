#!/usr/bin/env bash
# Print TEST_DATABASE_URL for a given Postgres hostname using the canonical
# defaults in docker/postgres/defaults.env. Prefer setting POSTGRES_TEST_HOST
# for pytest (see tests/postgres_defaults.py); use this when you need the
# literal URL string (docs, one-off tools).
#
# Usage:
#   ./docker/postgres/test-database-url.sh localhost   # host / venv pytest
#   ./docker/postgres/test-database-url.sh db-test     # CI compose service
#   ./docker/postgres/test-database-url.sh postgres    # app containers
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
set -a
source "${ROOT}/defaults.env"
set +a

HOST="${1:?usage: $0 <host>}"
echo "postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${HOST}:${POSTGRES_HOST_PORT}/${POSTGRES_TEST_DB}"
