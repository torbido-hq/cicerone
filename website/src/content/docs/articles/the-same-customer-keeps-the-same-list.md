---
title: The same customer keeps the same list
description: Cicerone materializes one top-K list per ranking recipe. Serve assigns each user to exactly one of those lists. The assignment is sticky. The contents of that list are not frozen.
date: 2026-09-08
excerpt: The job writes one list per recipe. Serve hashes the customer onto one of them. Sticky assignment is not a frozen top-K.
authors:
  - nicholas
---

You want half your signed-in traffic on one ranking recipe and half on a challenger. You write an `if` on the homepage. The same person comes back tomorrow and lands on the other side. You did not A/B test two ranking recipes. You randomized a page load.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) walkthrough already ends with a homemade split: hash `user_id`, bestsellers to one half, the personalized `SELECT` to the other. The instinct is right. A cookie rematches when it is cleared. A per-request coin flip rematches every load. That split is not “which model won this rank.” It is which recipe the customer is in.

Cicerone runs that split **offline**. The job writes **one top-K list per recipe** into the same recommendations table, stamped `variant`. Serve hashes the customer onto **exactly one** of those lists. The homepage `SELECT` does not pick an arm. Neither does a cookie.

```text
[job] + [experiment.variants]
        │
        ▼
   one top-K list per recipe
   (same table, column variant)
        │
        ▼
   GET /recommendations/{user_id}
        │  hash(experiment.id, user_id)
        ▼
   one list for that user
```

Four nouns, then stop mixing them.

| Noun | What it is |
| --- | --- |
| **Recipe** | The ranking configuration. Models, combiner, optional boosts. |
| **Variant list** | The materialized top-K rows that recipe wrote. |
| **Assignment** | Which of those lists this user gets. |
| **`source`** | Which model won **that rank** on **that** list. |

A treatment user can still have `popular_fallback` at rank 8. That is a hole on their list, not a flip onto the control recipe.

You get **one** `[experiment]` per config file. That is not a database lock. Old `experiment_id` values can still sit in track, exposures, and `experiment_state`.

## Before you enable experiments

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) post creates `unique (user_id, item_id)`. That is correct for **one** list. It is **wrong** for two.

`variant` is optional in the schema. Cicerone does **not** create a uniqueness constraint. The same `(user_id, item_id)` **will** appear twice, once per recipe, when those lists overlap. A host unique `(user_id, item_id)` rejects the overlapping rows. Cicerone writes every variant in one replace; that write fails and the previous table stays. Use `unique (user_id, item_id, variant)`.

If the column is missing, the job raises `RecommendationSchemaError`. Serve still returns rows and nulls `experiment_id` / `variant`. You are not running this experiment.

## Two recipes

```toml
[job]
models = ["als", "popular"]
top_k = 20

[experiment]
id = "homepage-v1"
enabled = true
primary_metric = "purchase"
attribution = "user"

[[experiment.variants]]
name = "control"
traffic = 0.5

[[experiment.variants]]
name = "blend"
traffic = 0.5
models = ["als", "bpr", "popular"]
combiner = "blend"
```

Control inherits `[job]` if you omit `models` / `combiner`. Traffic must be ≥ 0 and sum to at most 1; if it is below 1, the remainder goes to the last variant (see Reference).

`job.run()` unions models, fits once, then recommends once per recipe. It concatenates the frames and writes one table.

## How sticky assignment works

Alice and Bob both `GET /recommendations/{user_id}`. Same experiment. Different users. Each gets a deterministic bucket. As long as the experiment ID, traffic, and variant order stay the same, that bucket keeps resolving to the same variant. Clearing a cookie does not rematch them.

Serve does **not** take `?variant=`. It hashes:

```text
blake2s( f"{experiment.id}\0{user_id}".encode() , digest_size=8 )
int.from_bytes(..., "big") / 2**64     →  u ∈ [0, 1)
```

Then it walks the variants in **TOML order** and takes the first whose cumulative traffic is `> u`. The last name always gets leftover mass. Same inputs, same name, every request, every replica.

| You change | What happens |
| --- | --- |
| `experiment.id` | New digest. New assignment. |
| Traffic or **order** | Same digest. Different slice. Different list. |
| **Names only** | Same digest. Same slice. **New label.** Serve filters the new name. Last night’s rows miss until the next job. |
| Recipe knobs, same names | Same assignment. Next `job.run()` rewrites **that** list’s rows. |
| Promote | Everyone gets the winner’s name. Hash unused until Resume. |

Renaming is not a new experiment id. The bucket is the same. The name on the bucket is not. If you need a clean remap, change `id`.

Unknown users are split too. Serve hashes the **requested** `user_id`, then reads that variant’s `'__cold_start__'` rows. `GET /recommendations/__cold_start__` hashes the sentinel once, so it is one bucket. Cold-start is a **row**, not a second assignment rule.

## Sticky is not frozen

Assignment means Alice stays on `blend`. It does **not** mean last night’s twenty SKUs are immutable.

Tonight’s `job.run()` writes a new `blend` list. Alice is still `blend`. Rank 3 can be a different SKU. That is the point of a nightly job: the recipe stays, the catalog moves.

## How serve chooses the list

```http
GET /recommendations/alice
```

Bearer only if `[serve].auth_token` is set. `limit` / `k` and `category` apply **after** the assigned list is chosen. There is no `variant` query parameter. Assignment is serve’s job.

The nightly `SELECT … ORDER BY rank` does not assign. Either call serve, or keep the homemade split and do not call it this experiment. If you later turn `[experiment]` off, leftover variant rows are filtered, not deleted (Reference).

## How you know which list was better

Splitting traffic is easy. Knowing whether the **recipe** caused better purchases is the hard part.

This walkthrough uses `primary_metric = "purchase"` and `attribution = "user"`: a purchase from `[input]` counts for whoever that user is assigned to, whether or not they opened `/recommendations`. That measures the customer outcome directly. It is also noisy. The config default metric is `weighted`, not `purchase`.

| `attribution` | What it counts |
| --- | --- |
| `user` | Purchases (or clicks) from that user. No `/track`. |
| `recommended` | Same assignment. Event item must be on **that** list. Still no `/track`. |
| `click` | Needs `[track]`. Click, then purchase in the window (default 24h). |
| `impression` | Needs `[track]`. Impression, then purchase. GET is not an impression unless `log_impressions = true`. |

`ctr` / `conversion` need `[track]` and `click` / `impression`. `min_impressions` is 100 **impression rows**, not GET hits.

**Do this before you trust a week of `user` numbers:** set `log_exposures = true`. Without it, the dashboard hashes historical purchasers against **today’s** `id`, traffic, and order. Change any of those and Alice can move arms **in the report**. Serve did not rematch her live traffic. The evaluator rematched the CSV. With the flag on, the earliest `exposed_at` for this `experiment_id` pins her (a later-arriving row with an earlier stamp can replace the pin); events before that stamp are dropped. An empty exposure log assigns nobody.

The interval on the dashboard is a Robbins–Siegmund **mixture bound** on the mean difference, not a full anytime-valid confidence sequence and not a LIL CS. Alpha is split across non-control arms (`alpha / max(1, n−1)`). Peeking is the point. Do not quote a paper this binary does not implement.

## Ship the winner, or don’t

A week later `blend` wins. You click **Promote**. Tomorrow every `GET` reads the `blend` list. You did not paste TOML on the homepage.

Resume puts the hash back. The job does **not** copy `blend` into `[job]`. After you disable the experiment you still have whatever `[job]` always was, plus leftover variant rows.

Promote is refused when a CI is still undecided, two arms tie on the mean, a guardrail fails, the winner is already promoted, or `promoted_at` is set and unparsable. `ctr` / `conversion` also need enough impression rows (`min_impressions`). State lives in `experiment_state.json` or the `experiment_state` table (`experiment_id`, `promoted_variant`, `promoted_at`). Promote survives later jobs. If you rename the winner away, Promote’s name is gone and serve hashes again.

## What happens if someone checks out this afternoon

The [checkout](/articles/this-afternoons-checkout-can-move-the-row/) post can still flush popular / latest. The webhook does **not** rank the catalog.

Only the **assigned** (or promoted) variant is rewritten. The other list stays on last night’s job slice. Online LightFM is **not** started while the experiment is on (`Online collaborative refresh skipped while [experiment] is enabled`). Personalized ranks stay on the last `job.run()`.

That is write-through on one slice. It is not request-path inference.

## When this is the wrong tool

- You need a different rank **this request** — bandit, logged-out edge, geo. That is request-path ranking. Cicerone’s GET is a lookup.
- You need two overlapping experiments on the same user. One `[experiment]` per file.
- You wanted GrowthBook / Statsig: flags, holdouts, layers, warehouse assignment. Different product.
- You wanted a request-path bandit. Job-time Thompson exists behind `[track]` and `cicerone-recommender[bandits]`. Ship is still a 100% Promote. Not this walkthrough.
- You wanted a bestsellers `SELECT`. You do not need this table.

## After a week

Alice was `blend` on Monday and `blend` on Sunday. Tuesday’s job rewrote her twenty rows. Promote, if you clicked it, is a different sentence: everyone reads `blend`. Resume puts the hash back.

Change `id`, traffic, or order if you **mean** to rematch live assignment. Change names only if you mean to relabel a stable bucket. Change recipe knobs if you mean to rewrite that list and keep the people.

If `log_exposures` was off, the dashboard can still walk Alice to the other arm in a CSV when you edit today’s TOML. That is the report. It is not serve.

The customer keeps the same assignment. The assigned list does not keep last night’s SKUs.

The customer keeps the same list. The list does not keep last night’s SKUs.

<details>
<summary>Reference</summary>

The knobs and failure modes if you are wiring this up. The product page is [experiments](/experiments/).

**Traffic.** Names unique. At least two variants unless `automl_challenger = true`. Sum > 1 is `ConfigError`. Sum < 1: remainder is added to the **last** variant, plus a warning. `0.5` / `0.3` becomes `0.5` / `0.5`. `0.4` / `0.4` becomes `0.4` / `0.6`.

**Inheritance.** Combiner fallback: blend if blending is on, else RRF if `model_weights` is set, else priority. Boosts and eligibility default to inherit (`boosts` / `eligibility = true`). `false` drops them. A name list keeps a subset.

**Experiment off.** `resolve_assignment` is `(None, None)`. Promote is ignored. Leftover rows are **filtered** to `control` if that name exists, else `sorted(variant names)[0]`. Not a `DELETE`. Incremental follows the same rule when `variant_names == ()`.

**Guardrails (defaults).** Fallback-rate cap `0.5` (`popular_fallback` / `latest` / `incremental` and `+` mixes). Top-item share cap `0.4`. Distinct-item floor: `5`, or `min(5, max(1, catalog_size // 20))` if the catalog is known. Empty list fails. Missing `variant` column blocks Promote.

**Promote / Resume.** Dashboard Promote and Resume split. Missing winner name → hash again. Experiment disabled → leftover rows collapse as above.

**Failures.**

| What you did | What happens |
| --- | --- |
| Unique `(user_id, item_id)` | One replace fails; previous table stays. |
| No `variant` column | Job: `RecommendationSchemaError`. Serve: rows, null experiment fields. |
| `enabled = false`, rows remain | Filtered to `control` or `sorted(names)[0]`. |
| Rename after a job | Hash slice unchanged. New name has no rows until the next job. |
| Promote, then rename the winner | `promoted_variant` missing → hash again. |
| Thompson extra | Needs `[track]` + `cicerone-recommender[bandits]`. Not 100% ship. |

</details>
