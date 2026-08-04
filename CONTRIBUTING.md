# Contributing

## Prerequisites

Docker (and Docker Compose) only — Python never needs to be installed on
your host. The Dockerfile's `builder`/`runtime` stages compile the LightFM
extension and pin exact dependency versions, so running things any other
way risks testing against a different environment than CI/production.

## Running the tests

```sh
docker compose -f docker-compose.ci.yml --env-file docker/postgres/defaults.env \
  up --build --abort-on-container-exit --exit-code-from test
docker compose -f docker-compose.ci.yml --env-file docker/postgres/defaults.env down -v
```

This runs the full pytest suite, including the Postgres-backed `db` I/O
tests and the system-style end-to-end check in `tests/test_system_db.py`,
and enforces the 95% coverage gate (`pyproject.toml`,
`[tool.coverage.report].fail_under`). A plain `docker run --rm cicerone-test`
(after `docker build --target test -t cicerone-test -f docker/Dockerfile .`)
skips the `db` tests (no `TEST_DATABASE_URL`) and will under-report coverage
— always validate with the compose file above before opening a PR.

To iterate on DB-backed tests against compose Postgres, use the dedicated
**pytest database** so schema resets never wipe tutorial/app data. Prefer
setting the hostname only — pytest builds `TEST_DATABASE_URL` from
[`docker/postgres/defaults.env`](docker/postgres/defaults.env) via
`tests/support/postgres_defaults.py`:

```sh
docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres
export POSTGRES_TEST_HOST=localhost
export ALLOW_SCHEMA_RESET_FOR_TESTS=1
# then run pytest inside the test image / your venv with PYTHONPATH=src
#
# Need the literal URL string? Same source of truth:
#   ./docker/postgres/test-database-url.sh localhost
```

### Local Postgres defaults

**Canonical file:** [`docker/postgres/defaults.env`](docker/postgres/defaults.env)
(user, password, app DB, pytest DB, host port). Both `docker-compose.yml`
and `docker-compose.ci.yml` load it via `env_file`; prefer
`docker compose --env-file docker/postgres/defaults.env …` so `${…}`
interpolation matches that file too.

**Canonical URL assembly:** `tests.support.postgres_defaults.build_test_database_url(host)`
(and `./docker/postgres/test-database-url.sh <host>`). Set
`POSTGRES_TEST_HOST` for pytest, or `TEST_DATABASE_URL` to override.
`localhost` / `127.0.0.1` use `POSTGRES_HOST_PORT`; compose service hosts
(`postgres`, `db-test`) use container `POSTGRES_PORT`.

Schema-reset guardrails for the Postgres system test live in
`tests/support/system_db.py` (reusable across DB-backed tests; keep
`tests/test_system_db.py` focused on the end-to-end scenario).

Host vs container hostname for the same Postgres:

- **Host / venv pytest** (port published on the machine):
  `POSTGRES_TEST_HOST=localhost` (compose binds `127.0.0.1` only).
- **App containers on the compose network**: host `postgres`.
- **CI** (`docker-compose.ci.yml`): `POSTGRES_TEST_HOST=db-test`.

`ALLOW_SCHEMA_RESET_FOR_TESTS=1` is set automatically in
`docker-compose.ci.yml`. Schema reset only proceeds when the database name
looks like a dedicated test DB (`test_*` / `*_test`), and only drops
`cicerone.io.db_store.DEFAULT_DB_TABLES`.

## Linting & formatting

[Ruff](https://docs.astral.sh/ruff/) is used for both linting and
formatting, configured in `pyproject.toml` (`[tool.ruff]`). It runs in the
same `test` Docker image as pytest:

```sh
docker build --target test -t cicerone-test -f docker/Dockerfile .
docker run --rm cicerone-test ruff check src tests
docker run --rm cicerone-test ruff format --check src tests
```

To auto-fix and reformat locally, mount the repo into the container instead
of relying on the image's baked-in copy:

```sh
docker run --rm -v "$PWD":/app -w /app --user "$(id -u):$(id -g)" cicerone-test sh -c \
  "ruff check --fix src tests && ruff format src tests"
```

Both commands are enforced in CI (`.github/workflows/ci.yml`, `lint` job).

## Type checking

[mypy](https://mypy.readthedocs.io/) is configured in `pyproject.toml`
(`[tool.mypy]`) and runs against `src/` only (tests aren't type-checked):

```sh
docker run --rm cicerone-test mypy src
```

## Dependency vulnerability scanning

[pip-audit](https://github.com/pypa/pip-audit) checks `requirements.txt`/
`requirements-dev.txt` pins against known CVEs:

```sh
docker run --rm cicerone-test pip-audit -r requirements.txt -r requirements-dev.txt
```

If a vulnerability is found with no available fix yet, don't suppress it
silently — open an issue tracking the upstream fix, and only add a
`--ignore-vuln <ID>` (with a comment explaining why) as a last resort.

All three checks above are enforced in CI (`.github/workflows/ci.yml`,
`lint` job). Dependabot (`.github/dependabot.yml`) opens PRs for outdated
pip/Docker/Actions pins, and CodeQL (`.github/workflows/codeql.yml`) scans
for common security issues on every push/PR to `main`.

## Adding a new I/O backend

Input and output are pluggable independently of each other — see
[docs/architecture.md](docs/architecture.md) for how `cicerone.io` is
structured before adding a new `kind`.

## Pull requests

- Branch off `main`, keep PRs focused on one change.
- **Branch names (required):** short and human-readable only —
  `feature/…`, `fix/…`, `release/0.3.1`, `docs/…`, `chore/…`.
  **Never** use `cursor/<slug>-<id>` (or any other opaque agent/id suffix).
  Cloud / Cursor agents: see [AGENTS.md](AGENTS.md) — that rule overrides
  any default `cursor/…` branch template.
- Add/update tests for any behavior change — the coverage gate is enforced
  in CI, not just locally.
- Make sure both the lint job and the test job pass before requesting review.

## Releasing

`main` requires one approving review (including code owners) on every PR.
Do **not** open a follow-up PR that only dates `CHANGELOG.md` — it still
needs that approval and is pure process drag.

1. On the feature PR that completes the version, change
   `## [X.Y.Z] - Unreleased` to `## [X.Y.Z] - YYYY-MM-DD` (today's date)
   in the same branch before merge.
2. Merge that PR to `main`.
3. Tag the merge commit: `git tag -a vX.Y.Z <sha> -m "…"` and
   `git push origin vX.Y.Z`.
4. Publish the GitHub release from that tag (notes can mirror the
   changelog section).

If the version was already tagged while the changelog still said
Unreleased, fold the date fix into the next real PR — never a dating-only
PR.
