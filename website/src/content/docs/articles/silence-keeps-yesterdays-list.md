---
title: Silence keeps yesterday's list
description: A .NET worker upserts Cicerone's Kafka publish messages into SQL Server. A missing user is not a delete. An empty recommendations array is.
date: 2026-09-25
excerpt: The nightly job publishes one JSON message per user who still has rows. SQL Server keeps anyone the topic did not mention.
authors:
  - nicholas
---

You have a .NET storefront and SQL Server. Cicerone's database is on another network, and you do not want `HttpClient` on the homepage. You subscribe to `cicerone.recommendations` and treat a user who sent nothing tonight as a user with no recommendations. You deleted nobody. You kept yesterday's rows.

[Cicerone](https://cicerone.dev) 0.8 can publish after it writes the recommendations table. The Kafka key is `user_id`. The value is one JSON document. Your worker replaces that user's rows in SQL Server. The page only `SELECT`s. There is no recommendations SDK, and there is no Python in the request.

The [nightly table](/articles/a-nightly-table-next-to-your-orders/) post is the other shape: the app joins the table Cicerone just replaced. This post is the shop that cannot see that table.

```text
job.run()
  │
  ├─ write the recommendations table
  └─ produce one Kafka message per user in that write
        │  key = user_id
        ▼
   .NET worker
        │  delete + insert that user, then commit the offset
        ▼
   SQL Server
        │
        └─ the page: SELECT … ORDER BY rank
```

## What arrives

Turn the sidecar on. The batch job can publish with `[events]` off. Serve, if you run it, still reads `[output]`.

```toml
[publish]
enabled = true
kind = "kafka"

[publish.options]
bootstrap_servers = "${KAFKA_BOOTSTRAP_SERVERS}"
topic = "cicerone.recommendations"
```

That needs `pip install 'cicerone-recommender[kafka]'`. `bootstrap_servers` and `topic` are required.

After a successful write, the producer sends one JSON document per user it is publishing. The key is the UTF-8 `user_id`. Who is in that set depends on the write. The body looks like this:

```json
{
  "user_id": "alice",
  "message_id": "…",
  "recommendations": [
    {
      "user_id": "alice",
      "item_id": "sku-42",
      "rank": 1,
      "score": 0.91,
      "source": "popular"
    }
  ]
}
```

`message_id` is a SHA-256 hex digest of the UTF-8 bytes from `json.dumps({"user_id", "recommendations"}, separators=(",", ":"), sort_keys=True, allow_nan=False)`. `sort_keys` sorts object keys, not recommendation rows. Row order is the frame order. `reasons` and `variant` are inside that hash when the job wrote those columns, and they appear on a row only then. The same serialized list is the same id, including an older generation that happens to contain it. A different list is a different id. A NaN `score` is refused before produce, so that call publishes nothing. Nightly logs `Publish failed after successful write`. Incremental logs `Incremental publish failed after successful write`.

Store `message_id`. Do not recompute it. The same id is a duplicate body. It does not order generations.

## Two publishes

The nightly job and an incremental flush do not mean the same thing when a user is missing.

| Write | Who gets a message | `"recommendations": []` |
| --- | --- | --- |
| Nightly `job.run()` | Users who have rows in that write | Not sent. A user who dropped out of the table is silence. |
| Incremental replace | Users in that replace set | Only when that user has no rows left. Delete those SQL Server rows. A replaced user who still has rows gets a non-empty list. |

A flush that does not replace anyone publishes nothing. The prior list stays in Cicerone's table, and it stays in yours.

`__cold_start__` is a `user_id` like any other. If that sentinel has rows in the write, you get a message. Store it. An incremental clear of the sentinel is `"recommendations": []`. Silence is a user you already stored and the topic did not mention. That is not cold start.

Leave `[experiment]` off for this worker. With it on, one message holds every variant row that write contained. Fixed allocation is every named recipe. Thompson is the current pair. `variant` is the recipe name. The worker does not know which list Alice was assigned. [Serve](/articles/the-same-customer-keeps-the-same-list/) does. If any row has `variant`, log it, commit the offset, and do not insert.

## The worker

One consumer group owns the SQL table. The group id is yours. Cicerone does not set one. A second group writing the same table will race. The key keeps one user on one partition, so this group sees that user in offset order.

The nightly job and the serve process are two producers. Set `job.trigger.lock_backend` to `postgres` or `redis`. The default `in_process` is not visible to serve, so an incremental clear and a delayed nightly list for the same user can be appended in either order. `ApplyAsync` treats any new `message_id` as the latest body, so a clear followed by an older non-empty list puts the rows back. With postgres or redis, serve skips writes while the job holds the lock, and the job publishes before it releases it. A consumer that does not rewind then sees the nightly messages first. A rewind can still replay an older body and undo a clear. This payload has no generation timestamp, so the worker cannot reject that replay.

Commit the offset after the SQL transaction commits. `EnableAutoCommit` stays false. The same `message_id` is a no-op, then you still commit the offset. A body that does not deserialize, a missing `recommendations` array, or any row with `variant` is parked: log it and commit the offset. Do not insert those rows. A truncation, a null, or a duplicate `(user_id, item_id)` is the same park. A deadlock or a dropped connection is not. That offset stays uncommitted, the process exits, and the next start reads the record again. Shutdown cancels `Consume`. The in-flight SQL transaction rolls back, and that offset is not committed.

`Confluent.Kafka` and `Microsoft.Data.SqlClient`. .NET 8.

```csharp
using System.Text.Json;
using System.Text.Json.Serialization;
using Confluent.Kafka;
using Microsoft.Data.SqlClient;

var bootstrap = Environment.GetEnvironmentVariable("KAFKA_BOOTSTRAP_SERVERS")
    ?? throw new InvalidOperationException("KAFKA_BOOTSTRAP_SERVERS");
var topic = Environment.GetEnvironmentVariable("CICERONE_RECOMMENDATIONS_TOPIC")
    ?? "cicerone.recommendations";
var sql = Environment.GetEnvironmentVariable("STOREFRONT_SQL")
    ?? throw new InvalidOperationException("STOREFRONT_SQL");

using var consumer = new ConsumerBuilder<string, string>(new ConsumerConfig
{
    BootstrapServers = bootstrap,
    GroupId = "storefront-recommendations",
    AutoOffsetReset = AutoOffsetReset.Earliest,
    EnableAutoCommit = false,
}).Build();
consumer.Subscribe(topic);

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

while (!cts.IsCancellationRequested)
{
    ConsumeResult<string, string> record;
    try
    {
        record = consumer.Consume(cts.Token);
    }
    catch (OperationCanceledException)
    {
        break;
    }
    RecommendationMessage? message;
    try
    {
        message = JsonSerializer.Deserialize<RecommendationMessage>(record.Message.Value);
    }
    catch (JsonException ex)
    {
        Console.Error.WriteLine($"park {record.TopicPartitionOffset}: {ex.Message}");
        consumer.Commit(record);
        continue;
    }
    if (message is null || message.Recommendations is null)
    {
        Console.Error.WriteLine($"park {record.TopicPartitionOffset}: missing recommendations");
        consumer.Commit(record);
        continue;
    }
    if (message.Recommendations.Exists(row => row.Variant is not null))
    {
        Console.Error.WriteLine($"park {message.UserId}: variant rows belong to serve assignment");
        consumer.Commit(record);
        continue;
    }
    try
    {
        if (!await ApplyAsync(sql, message, cts.Token))
        {
            consumer.Commit(record);
            continue;
        }
    }
    catch (OperationCanceledException)
    {
        break;
    }
    consumer.Commit(record);
}

static async Task<bool> ApplyAsync(string sql, RecommendationMessage message, CancellationToken ct)
{
    await using var connection = new SqlConnection(sql);
    await connection.OpenAsync(ct);
    await using var tx = (SqlTransaction)await connection.BeginTransactionAsync(ct);
    try
    {
        return await ApplyInTransactionAsync(connection, tx, message, ct);
    }
    catch (SqlException ex) when (ex.Number is 515 or 2601 or 2627 or 8152)
    {
        Console.Error.WriteLine($"park {message.UserId}: SQL {ex.Number} {ex.Message}");
        return false;
    }
}

static async Task<bool> ApplyInTransactionAsync(
    SqlConnection connection,
    SqlTransaction tx,
    RecommendationMessage message,
    CancellationToken ct)
{
    await using (var current = new SqlCommand(
        "SELECT message_id FROM cicerone_recommendation_messages WHERE user_id = @user_id",
        connection, tx))
    {
        current.Parameters.AddWithValue("@user_id", message.UserId);
        var applied = (string?)await current.ExecuteScalarAsync(ct);
        if (applied == message.MessageId)
        {
            await tx.CommitAsync(ct);
            return true;
        }
    }

    await using (var delete = new SqlCommand(
        "DELETE FROM cicerone_recommendations WHERE user_id = @user_id",
        connection, tx))
    {
        delete.Parameters.AddWithValue("@user_id", message.UserId);
        await delete.ExecuteNonQueryAsync(ct);
    }

    foreach (var row in message.Recommendations)
    {
        await using var insert = new SqlCommand(
            """
            INSERT INTO cicerone_recommendations (user_id, item_id, rank, score, source)
            VALUES (@user_id, @item_id, @rank, @score, @source)
            """,
            connection, tx);
        insert.Parameters.AddWithValue("@user_id", message.UserId);
        insert.Parameters.AddWithValue("@item_id", row.ItemId);
        insert.Parameters.AddWithValue("@rank", row.Rank);
        insert.Parameters.AddWithValue("@score", row.Score);
        insert.Parameters.AddWithValue("@source", row.Source);
        await insert.ExecuteNonQueryAsync(ct);
    }

    await using (var mark = new SqlCommand(
        """
        MERGE cicerone_recommendation_messages AS t
        USING (SELECT @user_id AS user_id) AS s
        ON t.user_id = s.user_id
        WHEN MATCHED THEN UPDATE SET message_id = @message_id
        WHEN NOT MATCHED THEN INSERT (user_id, message_id) VALUES (@user_id, @message_id);
        """,
        connection, tx))
    {
        mark.Parameters.AddWithValue("@user_id", message.UserId);
        mark.Parameters.AddWithValue("@message_id", message.MessageId);
        await mark.ExecuteNonQueryAsync(ct);
    }

    await tx.CommitAsync(ct);
    return true;
}

sealed record RecommendationMessage(
    [property: JsonPropertyName("user_id")] string UserId,
    [property: JsonPropertyName("message_id")] string MessageId,
    [property: JsonPropertyName("recommendations")] List<RecommendationRow> Recommendations);

sealed record RecommendationRow(
    [property: JsonPropertyName("user_id")] string UserId,
    [property: JsonPropertyName("item_id")] string ItemId,
    [property: JsonPropertyName("rank")] int Rank,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("source")] string Source,
    [property: JsonPropertyName("variant")] string? Variant);
```

An empty `recommendations` array takes the delete branch and writes no items. That is the incremental clear. The nightly job does not send that array for a user it dropped. A truncation, a null, or a duplicate key returns before `CommitAsync`. Disposing that transaction rolls the delete and the inserts back. The offset is what gets committed.

```sql
CREATE TABLE cicerone_recommendations (
    user_id nvarchar(128) NOT NULL,
    item_id nvarchar(128) NOT NULL,
    rank int NOT NULL,
    score float NOT NULL,
    source nvarchar(64) NOT NULL,
    CONSTRAINT pk_cicerone_recommendations PRIMARY KEY (user_id, item_id)
);

CREATE TABLE cicerone_recommendation_messages (
    user_id nvarchar(128) NOT NULL PRIMARY KEY,
    message_id char(64) NOT NULL
);
```

Two `nvarchar(128)` key columns are 512 bytes, under SQL Server's 900-byte index limit. Do not widen them past that limit. A value that does not fit is SQL 8152. The worker parks that offset. SQL 515 is a null. SQL 2601 and 2627 are a duplicate key. SQL 1205, a timeout, or a dropped connection is not parked.

The page reads the marker and the rows in one statement. Two selects can straddle a commit: the first can see no marker and the second can see zero rows, and the page would treat a clear as never seen.

```sql
SELECT m.message_id, r.item_id, r.rank, r.score, r.source
FROM cicerone_recommendation_messages AS m
LEFT JOIN cicerone_recommendations AS r ON r.user_id = m.user_id
WHERE m.user_id = @userId
ORDER BY r.rank;
```

No result means you have never applied a message for that user, so read `__cold_start__`. If the sentinel itself has no marker, or its marker is a clear, show nothing. A result whose `item_id` is null means the latest applied message was an explicit clear, so show nothing. Rows in the result are the last body this worker applied. They are yesterday's list only when no later offset has been applied. They can be older than Cicerone's table.

This page does not copy Serve. A clear stays empty. `GET /recommendations/{user_id}` substitutes `__cold_start__` when the user has no rows, and returns 404 if the sentinel is empty too.

## After the write

Publish runs only when the recommendations write succeeded and this run is still the latest manifest. A newer `generated_at` logs `Skipping publish: recommendations were superseded` (incremental: `Skipping incremental publish: recommendations were superseded`). If the manifest cannot be confirmed, nothing is produced. Nightly logs `Skipping publish: could not confirm sidecar generation`. Incremental logs `Skipping incremental publish: could not confirm sidecar generation`.

Connect, delivery, timeout, and a non-JSON score are logged and left there:

```text
Publish failed after successful write
Incremental publish failed after successful write
```

The job stays successful when publish logs one of those lines. `apply` returns, and the incremental worker acks the flush. The recommendation write is already done. A timeout or a delivery error can leave earlier users from that flush on the topic. A NaN score cannot. SQL Server can sit on the previous message for anyone this flush did not land.

`LockLostError` and `WriterLockBusyError` during publish are not those log lines. They leave `apply`, and the incremental worker nacks the batch. The recommendation write is not rolled back. On the nightly job, `LockLostError` during publish is re-raised, the manifest is marked failed, and the table write stands.

Operator detail is in [incremental events](/incremental-events/).

## When you should not do this

You share Cicerone's database. Join it. A nightly replace drops the user in the same write. This topic will not.

The storefront can call `GET /recommendations/{user_id}`. That response is the current list, or `__cold_start__` when the user has no rows. If the sentinel is empty too, the status is 404. That fallback is Serve. The SQL page above stays empty after a clear. You do not keep a second copy, and you do not invent a delete rule.

You turned `[experiment]` on. Assignment is a hash in serve. This JSON is every variant row that write contained.

You wanted the topic to be the source of truth. It is a copy, produced after the table, skipped when the generation is stale, and incomplete when produce fails.

## In the morning

Alice had twenty rows last night. Tonight's job did not put her in the frame. Kafka said nothing. SQL Server still has the twenty.

Bob's incremental flush cleared him. His message has `"recommendations": []`. Those rows are gone.

Silence keeps yesterday's list. The empty array is the clear, and the nightly publish does not send it for someone who disappeared.
