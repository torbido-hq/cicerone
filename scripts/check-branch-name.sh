#!/usr/bin/env bash
# Fail if the current / PR branch name violates CONTRIBUTING.md.
# Allowed: feature|fix|docs|chore|release|dependabot/...
# Forbidden: cursor/<slug>-<id> and other opaque agent suffixes.
set -euo pipefail

branch="${1:-${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}}"
if [[ -z "${branch}" ]]; then
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
fi

if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
  echo "branch-name check: could not determine branch name" >&2
  exit 1
fi

# Pushes to main are fine (merge commits / direct main CI).
if [[ "${branch}" == "main" ]]; then
  echo "branch-name check: ok (${branch})"
  exit 0
fi

# Dependabot uses its own prefix; keep allowing it.
if [[ "${branch}" == dependabot/* ]]; then
  echo "branch-name check: ok (${branch})"
  exit 0
fi

allowed='^(feature|fix|docs|chore|release)/[A-Za-z0-9][A-Za-z0-9._/-]*$'
if [[ ! "${branch}" =~ ${allowed} ]]; then
  cat >&2 <<EOF
branch-name check FAILED: '${branch}'

CONTRIBUTING.md requires short human-readable names only:
  feature/…, fix/…, docs/…, chore/…, release/0.3.1

Never use cursor/<slug>-<id> (or any opaque agent/id suffix), even if a
Cloud/Cursor default template suggests that shape.

Rename the branch and open/update the PR from the new name.
EOF
  exit 1
fi

# Extra hard-fail for the common agent template even if someone nests it.
if [[ "${branch}" == cursor/* || "${branch}" =~ -[0-9a-f]{4}$ ]]; then
  cat >&2 <<EOF
branch-name check FAILED: '${branch}' looks like an opaque agent branch
(cursor/… or -<4 hex> suffix). Use feature/… (etc.) per CONTRIBUTING.md.
EOF
  exit 1
fi

echo "branch-name check: ok (${branch})"
