# Agent notes (read this first)

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
