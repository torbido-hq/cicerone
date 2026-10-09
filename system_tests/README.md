# Robot Framework system tests

Black-box tests against the real `recommender`/`serve`/`dashboard`
containers over HTTP — not in-process like the pytest suite (see
`tests/test_system_db.py` for the in-process DB round-trip, and
`tests/test_serve.py` / `tests/test_dashboard.py` / `tests/test_trigger.py`
for the same endpoints via FastAPI's `TestClient`). Local-only for now, not
wired into CI.

```sh
docker compose -f docker-compose.robot.yml up --build -d \
  postgres recommender serve dashboard
docker compose -f docker-compose.robot.yml run --rm --build robot
docker compose -f docker-compose.robot.yml down -v
```

(`seed`/`robot` are one-shot containers that exit 0 on success — kept out
of the `up`/`--abort-on-container-exit` combo, which would otherwise treat
any container exiting, including a successful one-shot, as a stack failure.)

Results (`log.html`, `report.html`, `output.xml`) land in
`system_tests/results/`.

## Layout

- `resources/common.resource` — shared sessions/keywords/base URLs (all
  overridable via environment variables; defaults match the in-network
  compose service names/ports).
- `seed_catalog.py` — seeds a tiny events/users/items catalog into Postgres
  before `recommender` runs its first batch job. Keep user ids (`robot-u1`
  etc.) in sync with `resources/common.resource` if you change it.
- `serve.robot` / `dashboard.robot` / `trigger_retrain.robot` — one suite per
  service (health/auth smoke checks + a few functional assertions), plus a
  cross-container check that a webhook-triggered retrain shows up on the
  dashboard.

## Adding a suite

Point it at the running stack via `resources/common.resource`, and prefer
asserting on stable hooks already used by the pytest suite (e.g.
`data-run-status="..."` on the dashboard) over fragile CSS/text matching.
