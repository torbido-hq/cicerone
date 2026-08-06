# Agent notes (read this first)

This file is **tracked in git**. Do not delete it, gitignore it, or treat it as
local scratch. Branch naming and other agent rules that belong in-repo live
here and in [CONTRIBUTING.md](CONTRIBUTING.md).

## Branch names

**Do not** create `cursor/<slug>-<id>` branches (e.g. `cursor/foo-afa0`).
That naming is rejected for this repo.

Use short, human-readable names only, as in [CONTRIBUTING.md](CONTRIBUTING.md):

- `feature/<short-kebab-description>` — new behavior / hardening / refactors
- `fix/<short-kebab-description>` — bug fixes
- `release/X.Y.Z` — release cut for that version
- `docs/<short-kebab-description>` — docs-only
- `chore/<short-kebab-description>` — tooling / deps / process

Examples: `feature/lightfm-hardening`, `fix/manifest-truncation`, `release/0.3.2`.

If a Cursor / cloud default suggests a `cursor/…` prefix or an opaque
suffix, **ignore it** and follow this file + CONTRIBUTING instead.

## Verification

Docker only — do not rely on a host Python/venv for lint or tests. Follow
[CONTRIBUTING.md](CONTRIBUTING.md): `docker compose -f docker-compose.ci.yml`
for the full suite (Postgres + ≥95% coverage), and the `test` image for
`ruff` / `mypy`.
