# Contributing

## Prerequisites

Docker (and Docker Compose) only — Python never needs to be installed on
your host. The Dockerfile's `builder`/`runtime` stages compile the LightFM
extension and pin exact dependency versions, so running things any other
way risks testing against a different environment than CI/production.

## Running the tests

```sh
docker compose -f docker-compose.ci.yml up --build --abort-on-container-exit --exit-code-from test
docker compose -f docker-compose.ci.yml down -v   # clean up the throwaway Postgres
```

This runs the full pytest suite, including the Postgres-backed `db` I/O
tests, and enforces the 95% coverage gate (`pyproject.toml`,
`[tool.coverage.report].fail_under`). A plain `docker run --rm cicerone-test`
(after `docker build --target test -t cicerone-test -f docker/Dockerfile .`)
skips the `db` tests (no `TEST_DATABASE_URL`) and will under-report coverage
— always validate with the compose file above before opening a PR.

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
- Add/update tests for any behavior change — the coverage gate is enforced
  in CI, not just locally.
- Make sure both the lint job and the test job pass before requesting review.
