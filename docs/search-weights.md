<img src="../src/cicerone/static/cicerone-logo.svg" alt="Cicerone" width="200">

# Search weights

Cicerone trains **user–item** lists. Meilisearch and OpenSearch rank **query–document** hits. They do not share a model. After each successful job, Cicerone writes catalog-wide scores those engines can store on documents. Serve stays a lookup.

`item_scores` is not truncated to `[job].top_k` and is not copied from the recommendations table. Incremental events do **not** rewrite it. Popularity on the search index lags until the next `job.run()` (cron or `POST /trigger/retrain`).

Personalized search still uses `GET /recommendations/{user_id}`. Do not put per-user fields in the search index.

## Output

Every successful job writes `item_scores` next to recommendations:

| column | definition |
| --- | --- |
| `item_id` | catalog id |
| `popular_score` | `sum(weight)` of `build_interactions` for that item |
| `latest_score` | same, on events in the last 14 days (`LATEST_WINDOW_DAYS`) |
| `n_users` | distinct users in the full (non-window) interactions |

Universe: `items.item_id` (when present) union items that appear in interactions. Catalog-only items get `0`. Dataset path: `item_scores.parquet`. DB: table `item_scores` (`[output.options].item_scores_table`).

## Serve

`GET /item-scores` (same bearer as recommendations):

| query | default | notes |
| --- | --- | --- |
| `limit` | 1000 | cap 5000 |
| `cursor` | _(none)_ | seek after this `item_id` |
| `item_id` | _(none)_ | one document |

```json
{
  "items": [
    {"item_id": "ipa-001", "popular_score": 12.4, "latest_score": 3.1, "n_users": 8}
  ],
  "next_cursor": "ipa-001"
}
```

`ServeClient.item_scores()` returns the same document. Missing `item_id` on a single-id request yields an empty `items` list.

## Meilisearch (host indexer)

Copy scores onto every document. Missing attributes make custom ranking undefined — write `0`.

Numeric fields `cicerone_popular`, `cicerone_latest`. Custom rule **after** text:

```text
["words", "typo", "proximity", "attribute", "sort", "exactness", "cicerone_popular:desc"]
```

Logged-in: RRF the Meili hit list with Cicerone ranks (`weight / (rrf_k + rank)`), or a second `filter` search merged. No per-user fields in the index.

## OpenSearch (host indexer)

Map `cicerone_popular` / `cicerone_latest` as `float`. Query with `function_score` + `field_value_factor` `log1p` on `cicerone_popular`, `missing: 0`, `boost_mode: multiply`.

Logged-in: extra `filter` function on recommended ids, or `should` `term` boosts from serve `score`.

## What this is not

- A Meilisearch or OpenSearch client
- Incremental rewrite of catalog popularity
- OpenSearch LTR / learned BM25
- Changing Meili `_rankingScore` (custom rules do not affect it)
