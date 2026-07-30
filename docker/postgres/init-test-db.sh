#!/bin/bash
# Creates the pytest database on first boot of the compose postgres service.
# Values come from docker/postgres/defaults.env (via the service env_file).
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" <<-EOSQL
	CREATE DATABASE ${POSTGRES_TEST_DB} OWNER ${POSTGRES_USER};
EOSQL
