#!/bin/bash
# Creates the pytest database on first boot of the compose postgres service.
# Values come from docker/postgres/defaults.env (via the service env_file).
set -euo pipefail

# Match cicerone.io.options.sql_identifier / looks_like_test_database so a
# crafted compose/env override cannot inject statements into CREATE DATABASE.
_ident_re='^[A-Za-z_][A-Za-z0-9_]*$'
for var in POSTGRES_USER POSTGRES_DB POSTGRES_TEST_DB; do
  val="${!var:-}"
  if [[ ! "${val}" =~ ${_ident_re} ]]; then
    echo "error: ${var}=${val@Q} must be a simple SQL identifier ([A-Za-z_][A-Za-z0-9_]*)" >&2
    exit 1
  fi
done
if [[ ! "${POSTGRES_TEST_DB}" =~ ^test_ ]] && [[ ! "${POSTGRES_TEST_DB}" =~ _test$ ]]; then
  echo "error: POSTGRES_TEST_DB=${POSTGRES_TEST_DB@Q} must start with 'test_' or end with '_test'" >&2
  exit 1
fi

# Quoted heredoc + psql :"ident" vars — no shell interpolation into SQL text.
psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" \
  -v test_db="${POSTGRES_TEST_DB}" \
  -v db_owner="${POSTGRES_USER}" <<'EOSQL'
	CREATE DATABASE :"test_db" OWNER :"db_owner";
EOSQL
