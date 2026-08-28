<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Experiments

Cicerone tests **whole ranking recipes** (models + combiner + optional blending
knobs and boost/eligibility policy), not which `source` in a mixed cascade got
the click. The unit of assignment is a user. Serve stays a lookup: the job
writes extra recommendation rows tagged with `variant`, and
`GET /recommendations/{user_id}` hashes the user onto one of those lists.

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
# boosts = true               # inherit [[boost]] from features.toml
# boosts = false              # drop all boosts
# boosts = ["featured"]       # named subset of features.toml
# [[experiment.variants.boost]]
# name = "new-arrivals"       # replacement rules for this variant only
# kind = "boolean"
# item_column = "is_new"
# factor = 1.3
# eligibility = true
```

Validation: unique names; traffic ≥ 0 and sums to at most 1 (any remainder
is assigned to the last variant). Recipe overrides are the knobs `[job]`
already has, plus optional `boosts` / `eligibility` (inherit, drop, named
subset, or replacement `[[experiment.variants.boost]]` /
`[[experiment.variants.eligibility]]` tables).

## Assignment

Sticky, replica-safe **as long as** `experiment_id`, variant **names**,
**order**, and `traffic` stay fixed: `blake2s(experiment_id || "\0" || user_id)`
→ `[0, 1)`, then walk cumulative `traffic`. Changing traffic remaps users.
A dashboard **Promote** writes `experiment_state` next to the output store;
serve then sends 100% traffic to that variant until you **Resume split**
(or delete promote state / turn the experiment off). Promote survives later jobs. After you disable
`[experiment]`, leftover `variant` rows collapse to `control` (else the
lexicographically first remaining name; blank/NaN names are ignored) on
**serve reads and incremental write-through** until the next job rewrite.
Do not mix lists.

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
variant's `__cold_start__` rows. Prometheus
`cicerone_recommendations_experiment_total{experiment_id,variant}` counts
lookups.

`[experiment].log_exposures = true` appends JSONL (`exposures.jsonl`) on a
**local** dataset path, or a `recommendation_exposures` DB table. Object-store
(S3) JSONL append is refused: it is not atomic. With `events.ha = true`,
exposures require **db** output (two replicas must not append the same local
file). Default off: serve stays read-only.

Impression and click tracking (CTR / conversion of shown items) is a separate
`POST /track` contract. See [evaluation.md](evaluation.md). Do not send
impressions through `POST /events`.

## Incremental events

Popular/latest write-through refreshes **every variant** for affected users.
`[events.online]` LightFM rewrite is **skipped** while `[experiment]` is on
(so arms stay isolated; load/serve log a warning). Without an experiment, it
rewrites personalized / item-KNN / content-fallback rows for affected users.
See [incremental-events.md](incremental-events.md).

## Recipe vs measurement

Per-variant `boosts` / `eligibility` change **what is on the served list**.
That is the recipe half of 0.8 evaluation: you can A/B a merchandising
policy (drop boosts, keep `["featured"]`, or ship a replacement
`[[experiment.variants.boost]]` table) without cloning the whole combiner.

0.7 experiment metrics still join **all** `[input]` events to the hashed
variant (intention-to-treat). A purchase that came from search or email is
not evidence that the treatment list worked, so you cannot honestly A/B
those lists on purchases alone.

Knowing whether recommendations are working needs **served-item** signals
(impressions and clicks on ranks from `GET /recommendations`). That ingest
is **not** `POST /events` and must not enter `[event_weights]` / LightFM.
Until that measurement layer lands on this train, policy variants can still
run; do not treat purchase ITT as CTR.

## Metrics and promote

Deterministic assignment means ingested `[events]` join a variant without a
new impression protocol when `attribution = "user"`. Counts and weighted sums
(`features.toml` `[event_weights]`) are **descriptive** until the sequential
test decides.

To A/B the lists themselves, set `primary_metric = "ctr"` or `"conversion"`
and `attribution = "click"` or `"impression"` after wiring `[track]`.
`attribution = "recommended"` counts only events whose item was on the
assigned list (no `/track`). See [evaluation.md](evaluation.md).

The dashboard **Experiments** page (`GET /dashboard/experiments`) shows:

- Approximate always-valid CIs on the primary metric (a LIL-style radius for
  peeking — not a full anytime-valid CS for heavy-tailed outcomes).
  Intention-to-treat on users who appear in the event window, or
  exposure-conditional when `log_exposures` is on (rows must match this
  `experiment_id`).
- Guardrails (fail closed): fallback rate, top-item share, distinct-item
  coverage against the **items snapshot** size (not recommended-item
  diversity). Missing recommendations or a `variant` column also block
  promote. A purchase win that collapses the catalog does not promote.
- **Promote** — only when the CI excludes zero **and** guardrails pass —
  writes the winner as 100% traffic at serve. **Resume split** clears
  promote state so sticky hashing resumes. The next job can copy that
  recipe into `[job]` and disable the experiment.

`automl_challenger = true` (with `[job.automl]` enabled) synthesizes
`control` from the last successful manifest (the `experiment_variants`
control recipe when present, not the union `models` list) and `treatment`
from this run's AutoML pick. Incremental write-through uses
`control`/`treatment` even when `[[experiment.variants]]` is empty.

## What this is not

- Live LightFM / SASRec in `GET /recommendations`
- Per-source positive-feedback-rate of a mixed cascade
- Multi-layer concurrent experiments
