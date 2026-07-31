#!/usr/bin/env bash
# Print TEST_DATABASE_URL for a given Postgres hostname using the canonical
# defaults in docker/postgres/defaults.env. Prefer setting POSTGRES_TEST_HOST
# for pytest (see tests/support/postgres_defaults.py); use this when you need the
# literal URL string (docs, one-off tools).
#
# Process-environment POSTGRES_* values win over defaults.env (same as
# tests/support/postgres_defaults._default), so remapped host ports stay in sync.
#
# localhost / 127.0.0.1 / ::1 → POSTGRES_HOST_PORT (published map)
# any other host (postgres, db-test) → POSTGRES_PORT (container listen port)
#
# Usage:
#   ./docker/postgres/test-database-url.sh localhost   # host / venv pytest
#   ./docker/postgres/test-database-url.sh db-test     # CI compose service
#   ./docker/postgres/test-database-url.sh postgres    # app containers
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULTS_FILE="${ROOT}/defaults.env"

# Load file defaults only for variables not already set in the environment.
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%%#*}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "${line}" || "${line}" != *=* ]] && continue
  key="${line%%=*}"
  value="${line#*=}"
  if [[ -z "${!key+x}" ]]; then
    printf -v "${key}" '%s' "${value}"
    export "${key?}"
  fi
done < "${DEFAULTS_FILE}"

HOST="${1:?usage: $0 <host>}"
case "${HOST}" in
  localhost|127.0.0.1|::1) PORT="${POSTGRES_HOST_PORT}" ;;
  *) PORT="${POSTGRES_PORT}" ;;
esac

case "${POSTGRES_TEST_DB}" in
  test_*|*_test) ;;
  *)
    echo "POSTGRES_TEST_DB must look like a dedicated test database (start with 'test_' or end with '_test'), got '${POSTGRES_TEST_DB}'" >&2
    exit 1
    ;;
esac

echo "postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${HOST}:${PORT}/${POSTGRES_TEST_DB}"
