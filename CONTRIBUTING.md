# Contributing

## Prerequisites

Docker (and Docker Compose) only — Python never needs to be installed on
your host. The Dockerfile's `builder`/`runtime` stages compile the LightFM
extension and pin exact dependency versions, so running things any other
way risks testing against a different environment than CI/production.

## Running the tests

```sh
docker compose -f docker-compose.ci.yml --env-file docker/postgres/defaults.env \
  up --build --abort-on-container-exit --exit-code-from test test
docker compose -f docker-compose.ci.yml --env-file docker/postgres/defaults.env \
  --profile sequential run --rm --build test-sequential
docker compose -f docker-compose.ci.yml --env-file docker/postgres/defaults.env down -v
```

This runs the full pytest suite, including the Postgres-backed `db` I/O
tests and the system-style end-to-end check in `tests/test_system_db.py`,
and enforces the 95% coverage gate (`pyproject.toml`,
`[tool.coverage.report].fail_under`). `test-sequential` then runs the
SASRec/BERT4Rec/HSTU extra tests (`rectools[torch]`) in a separate image —
the main `test` and runtime images stay torch-free, because RecTools
imports its NN stack whenever torch is installed. Model/config tests
mirror the packages (`tests/test_model_*.py`, `tests/test_config_*.py`;
shared helpers under `tests/support/`). A plain `docker run --rm cicerone-test`
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
# then run pytest inside the test image / your venv
# (`pip install -e '.[redis]'` from a checkout, or PYTHONPATH=src)
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
`requirements-dev.txt` / `requirements-bandits.txt` pins against known CVEs:

```sh
docker run --rm cicerone-test pip-audit -r requirements.txt -r requirements-dev.txt -r requirements-bandits.txt
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

## Serve API contract

Changing the serve HTTP surface (`cicerone.serve` / `serve_schemas.py`)
means regenerating the checked-in OpenAPI document and keeping the thin
client in sync:

```sh
docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app/src cicerone-test \
  cicerone export-openapi -o docs/openapi/serve.openapi.json
```

`tests/test_serve_openapi_client.py` asserts the committed file matches
`create_app(...).openapi()`. Update `cicerone.serve_client.ServeClient`
and/or `examples/serve/` when request/response fields change.

## Pull requests

- Branch off the train you are targeting
  ([Release trains](#release-trains);
  [Current trains](.cursor/rules/releases.mdc#current-trains)).
  Keep PRs focused on one change.
- **Branch names (required):** short and human-readable only —
  `feature/…`, `fix/…`, `release/0.3.1`, `docs/…`, `chore/…`.
  **Never** use `cursor/<slug>-<id>` (or any other opaque agent/id suffix).
  Cloud / Cursor agents: follow this rule even if a default template
  suggests a `cursor/…` prefix.
- **Never** add `Made with Cursor` (or a cursor.com attribution link) to
  PR descriptions, commits, or comments.
- **Agent instructions:** Enforceable agent rules live in
  [`.cursor/rules/`](.cursor/rules/) (`alwaysApply: true`) and this file —
  that is what Cloud Agents actually see every session. A local `AGENTS.md`
  is gitignored and **cannot** enforce anything on Cloud. Extra personal
  notes (not repo policy): Cursor **User Rules** / **Team Rules**.
- Add/update tests for any behavior change — the coverage gate is enforced
  in CI, not just locally.
- Make sure the `ci` job passes before requesting review. Python changes
  also run `lint` and `test`. PRs that only touch `website/**` skip those
  Docker jobs; `ci` still succeeds. The Pages workflow builds the Starlight
  site (`cd website && npm ci && npm run build`).

## Release trains

[Current trains](.cursor/rules/releases.mdc#current-trains) is the
**single source of truth** for current train versions. Any change
to branch ownership must update that file in the same PR.

While a **patch** and a **minor** are in flight together, `main` is the
patch train and `release/X.Y.0` is the minor train.

| Train | Branch | What lands |
|---|---|---|
| **Patch** | `main` | Bug fixes, small refactors, improvements (perf/docs/DX), dependency bumps, tests for existing behavior. No new user-facing capability, config keys, models, or I/O `kind`s. |
| **Minor** | `release/X.Y.0` (long-lived; PRs target this branch, not `main`; see [Current trains](.cursor/rules/releases.mdc#current-trains)) | New features and new public surface. |

- Fixes, refactors, improvements → patch (`main`). Features → minor
  (`release/X.Y.0`).
- Split mixed work (refactor + feature) into two PRs: patch first, then
  feature on the minor train. If the refactor has no value without the
  feature, keep both on the minor train. Never land a feature on the
  patch train because a refactor is in the same diff.
- If still ambiguous, use the patch train unless the change adds
  user-facing capability.
- Do not open or merge feature PRs into `main` while the patch is in
  flight. Feature work stays on `release/X.Y.0` and reaches `main` only
  via [Merge-backs](.cursor/rules/releases.mdc#merge-backs) after the
  patch is tagged (not by retargeting those PRs at `main`).
- After the patch is tagged, follow
  [Merge-backs](.cursor/rules/releases.mdc#merge-backs) and update
  **Current trains** in that same PR.
- After dual-train ends, see
  [Single train](.cursor/rules/releases.mdc#single-train).

Procedures:
[Cherry-picks](.cursor/rules/releases.mdc#cherry-picks),
[Hotfixes](.cursor/rules/releases.mdc#hotfixes),
[Merge-backs](.cursor/rules/releases.mdc#merge-backs).

## Releasing

`main` requires one approving review (including code owners) on every PR.
Do **not** open a follow-up PR that only dates `CHANGELOG.md` — it still
needs that approval and is pure process drag.

The PyPI project is **`cicerone-recommender`** (`pip install
cicerone-recommender`; `import cicerone`). The name `cicerone` is a
different package. Optional extras `[redis]`, `[kafka]`, `[rabbitmq]`,
`[sequential]`, and `[bandits]` match `requirements-redis.txt` /
`requirements-kafka.txt` / `requirements-rabbitmq.txt` /
`requirements-sequential.txt` / `requirements-bandits.txt`.

One-time PyPI setup (before the first upload): create a GitHub Environment
named `pypi`, then a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/)
for `cicerone-recommender` — owner `torbido-hq`, repo `cicerone`, workflow
`publish.yml`, environment `pypi`. No API token.

1. On the PR that completes the version, put notes under
   `## [X.Y.Z] - YYYY-MM-DD` (today's date) in the same branch
   before merge, and set `cicerone.__version__`
   (`src/cicerone/__init__.py`) to the same `X.Y.Z` (serve OpenAPI
   metadata uses it via `SERVE_API_VERSION` — regenerate
   `docs/openapi/serve.openapi.json` if the version string changed).
2. Tagging always happens from `main`, even when the completing PR
   lands on the minor train. Never tag `release/X.Y.0`.
   - **Patch:** merge the completing PR to `main`. Tag that commit on
     `main`: `git tag -a vX.Y.Z <sha> -m "…"` and
     `git push origin vX.Y.Z`.
   - **Minor:** merge the completing PR to `release/X.Y.0`. Merge
     `release/X.Y.0` into `main`
     ([Merge-backs](.cursor/rules/releases.mdc#merge-backs)). Tag
     that commit on `main`: `git tag -a vX.Y.Z <sha> -m "…"` and
     `git push origin vX.Y.Z`.
3. Publish the GitHub release from that tag (notes can mirror the
   changelog section). `.github/workflows/publish.yml` builds the sdist and
   wheel (including dashboard CSS) and uploads them to PyPI.

If the version was already tagged while the changelog heading still
lacked a date, fold the date fix into the next real PR — never a dating-only
PR.
