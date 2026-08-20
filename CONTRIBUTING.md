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
`[tool.coverage.report].fail_under`). Model/config tests mirror the
packages (`tests/test_model_*.py`, `tests/test_config_*.py`; shared helpers
under `tests/support/`). A plain `docker run --rm cicerone-test`
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

- Branch off the train you are targeting (patch = `main`, minor =
  `release/X.Y.0`). Keep PRs focused on one change.
- **Branch names (required):** short and human-readable only —
  `feature/…`, `fix/…`, `release/0.3.1`, `docs/…`, `chore/…`.
  **Never** use `cursor/<slug>-<id>` (or any other opaque agent/id suffix).
  Cloud / Cursor agents: follow this rule even if a default template
  suggests a `cursor/…` prefix.
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

`release/X.Y.0` in this section is a **placeholder** for the in-flight
minor branch. **Current trains** is the only concrete mapping — substitute
those versions whenever you see `X.Y.0`. Who updates **Current trains**
and when is below that snippet.

While a **patch** and a **minor** are in flight together, `main` is the
patch train and `release/X.Y.0` is the minor train.

| Train | Branch | What lands |
|---|---|---|
| **Patch** | `main` | Bug fixes, small refactors, improvements (perf/docs/DX), dependency bumps, tests for existing behavior. No new user-facing capability, config keys, models, or I/O `kind`s. |
| **Minor** | `release/X.Y.0` (long-lived; PRs target this branch, not `main`) | New features and new public surface. |

- Fixes, refactors, improvements → patch (`main`). Features → minor
  (`release/X.Y.0`).
- Split mixed work (refactor + feature) into two PRs: patch first, then
  feature on the minor train. If the refactor has no value without the
  feature, keep both on the minor train. Never land a feature on the
  patch train because a refactor is in the same diff.
- If still ambiguous, use the patch train unless the change adds
  user-facing capability.
- Never land minor-train features on `main` until the patch is tagged.
- `## [Unreleased]` is per-branch (patch on `main`, minor on
  `release/X.Y.0`).
- After the patch is tagged: merge `release/X.Y.0` into `main`; then
  `main` is the minor train and new features land there. Stop
  cherry-picking. Update **Current trains** in that same PR (see
  below).

**Current trains** (concrete mapping; this line only):
patch **0.6.1** on `main`; minor **0.7.0** on `release/0.7.0`.

The PR that changes the split updates **Current trains** in the same
change (not a follow-up):

- After the patch tag, when merging `release/X.Y.0` into `main`: if you
  are not cutting a new split, set it to no split (`main` is the minor,
  using the version just tagged).
- When cutting a new patch+minor split, the PR that creates
  `release/X.Y.0` from `main` sets patch on `main` and minor on that
  branch.

Cherry-pick, hotfix, and merge-back commands are in
[Cherry-picks and merge-backs](#cherry-picks-and-merge-backs).

### Cherry-picks and merge-backs

Checklist (see **Current trains** for today's `X.Y.0`):

1. **Fix** — PR to `main`, then cherry-pick onto `release/X.Y.0`.
2. **Feature** — PR to `release/X.Y.0` (not `main`).
3. **Hotfix (both trains)** — PR to `main` first, cherry-pick immediately.
4. **Merge-back** — after the patch tag, cherry-pick any remaining patch
   commits onto `release/X.Y.0`, merge it into `main`, then tag.

Cherry-pick each patch-train fix onto `release/X.Y.0` after it lands on
`main` (the commit that landed the change on `main`):

```sh
git checkout release/X.Y.0
git cherry-pick <sha>
git push origin release/X.Y.0
```

If the cherry-pick conflicts, keep the minor-train feature code and
re-apply the patch fix on that surface. If it still will not apply (the
minor branch has a different API or config), abort
(`git cherry-pick --abort`) and open a small PR targeting
`release/X.Y.0` that ports the same fix. Do not change `main` to match
minor-only APIs.

**Example.** A dashboard lookup crash and a new serve filter in the same
week:

1. **Fix:** branch `fix/dashboard-lookup` from `main`, PR into `main`,
   then `git cherry-pick <sha>` onto `release/X.Y.0`.
2. **Feature:** branch `feature/serve-filters` from `release/X.Y.0`, PR
   into `release/X.Y.0` (not `main`).

**Hotfixes that must ship on both trains** always land on `main` first
(patch). Do not open the fix only against `release/X.Y.0`. After the
`main` PR merges, cherry-pick immediately (same commands as above) so the
minor train does not wait for the next routine backport.

When merging `release/X.Y.0` into `main` (after the patch tag, before
tagging the minor): if `main` has patch commits not yet on
`release/X.Y.0`, cherry-pick those first, then merge. Do not force-push
`main`. If a conflict is a missing cherry-pick, abort the merge, port
the fix, and retry.

If the merge still conflicts, keep patch behavior from `main` and minor
features from the release branch:

- **Prefer `main`:** both sides edited the dashboard lookup error path.
  `main` has the patch crash fix; `release/X.Y.0` still has the unfixed
  copy. Take `main`'s error handling, then restore any minor-only fields
  around it.
- **Prefer `release/X.Y.0`:** `main` has no serve-filter helper;
  `release/X.Y.0` added `feature/serve-filters`. Keep the new helper and
  call sites from the release branch.

## Releasing

`main` requires one approving review (including code owners) on every PR.
Do **not** open a follow-up PR that only dates `CHANGELOG.md` — it still
needs that approval and is pure process drag.

The PyPI project is **`cicerone-recommender`** (`pip install
cicerone-recommender`; `import cicerone`). The name `cicerone` is a
different package. Optional extras `[redis]` and `[sequential]` match
`requirements-redis.txt` / `requirements-sequential.txt`.

One-time PyPI setup (before the first upload): create a GitHub Environment
named `pypi`, then a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/)
for `cicerone-recommender` — owner `torbido-hq`, repo `cicerone`, workflow
`publish.yml`, environment `pypi`. No API token.

1. On the PR that completes the version, move `## [Unreleased]` notes
   into `## [X.Y.Z] - YYYY-MM-DD` (today's date) in the same branch
   before merge, and set `cicerone.__version__`
   (`src/cicerone/__init__.py`) to the same `X.Y.Z` (serve OpenAPI
   metadata uses it via `SERVE_API_VERSION` — regenerate
   `docs/openapi/serve.openapi.json` if the version string changed).
2. Get that version onto `main`, then tag **from `main`**:
   - **Patch:** merge the completing PR to `main`. Tag that commit on
     `main`: `git tag -a vX.Y.Z <sha> -m "…"` and
     `git push origin vX.Y.Z`.
   - **Minor:** merge the completing PR to `release/X.Y.0`. Merge
     `release/X.Y.0` into `main` (see
     [Cherry-picks and merge-backs](#cherry-picks-and-merge-backs) if
     `main` diverged). Tag that commit on `main`:
     `git tag -a vX.Y.Z <sha> -m "…"` and `git push origin vX.Y.Z`.
3. Publish the GitHub release from that tag (notes can mirror the
   changelog section). `.github/workflows/publish.yml` builds the sdist and
   wheel (including dashboard CSS) and uploads them to PyPI.

If the version was already tagged while the changelog still said
Unreleased, fold the date fix into the next real PR — never a dating-only
PR.
