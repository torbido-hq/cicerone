<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Experiments

Cicerone tests **whole ranking recipes** (models + combiner + optional blending
knobs), not which `source` in a mixed cascade got the click. The unit of
assignment is a user. Serve stays a lookup: the job writes extra recommendation
rows tagged with `variant`, and `GET /recommendations/{user_id}` hashes the
user onto one of those lists.

One experiment at a time. Overlapping layers, request-path bandits, and
GrowthBook/Statsig as a required dependency are out of scope.

## Config

```toml
[experiment]
enabled = true
id = "rrf-vs-blend-2026-08"
primary_metric = "purchase"   # event_type key, or "weighted"
# log_exposures = false       # serve appends (user, experiment, variant, generated_at)
# automl_challenger = false   # control = last successful manifest; treatment = AutoML pick
# alpha = 0.05                # always-valid CI level

[[experiment.variants]]
name = "control"
traffic = 0.5
# omitted recipe inherits [job] models / model_weights / rrf_k

[[experiment.variants]]
name = "treatment"
traffic = 0.5
models = ["collaborative", "item_based", "popular", "latest"]
combiner = "blend"            # priority | rrf | blend
# boosts = true
# eligibility = true
```

Validation: unique names; traffic ≥ 0 and sums to at most 1 (remainder is
dumped on the last variant). Recipe overrides are the knobs `[job]` already
has, plus optional `boosts = false` / `eligibility = false` to drop policy
rules for that variant.

## Assignment

Sticky, replica-safe **as long as** `experiment_id`, variant **names**,
**order**, and `traffic` stay fixed: `blake2s(experiment_id || "\0" || user_id)`
→ `[0, 1)`, then walk cumulative `traffic`. Changing traffic remaps users.
A dashboard **Promote** writes `experiment_state` next to the output store;
serve then sends 100% traffic to that variant until you clear promote state
or turn the experiment off. Promote survives later jobs. After you disable
`[experiment]`, leftover `variant` rows collapse to `control` (else the
lexicographically first name) on **serve reads and incremental write-through**
until the next job rewrite. Do not mix lists.

## Job

The job fits the **union** of variant models once, then combines/blends each
recipe into top-K and tags `variant`. One recommendations table. The run
manifest records `experiment_id` and `experiment_variants` (JSON recipes) so a
later AutoML challenger can use the last successful recipe as control.

Existing tables without a `variant` column still serve (no experiment). Adding
the column to an existing DB table is `ALTER TABLE … ADD COLUMN variant TEXT`.

## Serve

`GET /recommendations/{user_id}` hashes the user, filters rows, and returns
`experiment_id` + `variant` (`null` when experiments are off, or when the
recommendations table has no `variant` column). Cold-start uses that
variant’s `__cold_start__` rows. Prometheus
`cicerone_recommendations_experiment_total{experiment_id,variant}` counts
lookups.

`[experiment].log_exposures = true` appends JSONL (`exposures.jsonl`) on a
**local** dataset path, or a `recommendation_exposures` DB table. Object-store
(S3) JSONL append is refused: it is not atomic. With `events.ha = true`,
exposures require **db** output (two replicas must not append the same local
file). Default off: serve stays read-only.

## Incremental events

Popular/latest write-through refreshes **every variant** for affected users.
`[events.online]` LightFM rewrite is **skipped** while `[experiment]` is on
(so arms stay isolated). See [incremental-events.md](incremental-events.md).

## Metrics and promote

Deterministic assignment means ingested `[events]` join a variant without a
new impression protocol. Counts and weighted sums (`features.toml`
`[event_weights]`) are **descriptive** until the sequential test decides.

The dashboard **Experiments** page (`GET /dashboard/experiments`) shows:

- Approximate always-valid CIs on the primary metric (a LIL-style radius for
  peeking — not a full anytime-valid CS for heavy-tailed outcomes). ITT on
  users who appear in the event window, or exposure-conditional when
  `log_exposures` is on (rows must match this `experiment_id`).
- Guardrails (fail closed): fallback rate, top-item share, distinct-item
  coverage. A purchase win that collapses the catalog does not promote.
- **Promote** — only when the CI excludes zero **and** guardrails pass —
  writes the winner as 100% traffic at serve. The next job can copy that
  recipe into `[job]` and disable the experiment.

`automl_challenger = true` (with `[job.automl]` enabled) synthesizes
`control` from the last successful manifest and `treatment` from this run’s
AutoML pick.

## What this is not

- Live LightFM / SASRec in `GET /recommendations`
- Gorse-style per-source positive-feedback-rate of a mixed cascade
- Multi-layer concurrent experiments
