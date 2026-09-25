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

After a successful write, the producer sends one message per user. The key is the UTF-8 `user_id`. The body looks like this:

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

`message_id` is a SHA-256 of `{user_id, recommendations}`. The same list hashes to the same id. `reasons` and `variant` appear on a row only when the job wrote those columns. A NaN `score` is refused before produce: the flush publishes nothing, and the log line is `Publish failed after successful write`.

Store `message_id`. Do not recompute it.

## Two publishes

The nightly job and an incremental flush do not mean the same thing when a user is missing.

| Write | Who gets a message | `"recommendations": []` |
| --- | --- | --- |
| Nightly `job.run()` | Users who have rows in that write | Not sent. A user who dropped out of the table is silence. |
| Incremental replace | Users in that replace set | That user has no rows left. Delete those SQL Server rows. |

A flush that does not replace anyone publishes nothing. The prior list stays in Cicerone's table, and it stays in yours.

`__cold_start__` is a `user_id` like any other. If that sentinel is in the write, you get a message. Store it. The page reads it when the signed-in user has no rows yet. Silence is not that case. Silence means you already stored them.

Leave `[experiment]` off for this worker. With it on, one message can hold every recipe's rows for that user, and `variant` is how you tell them apart. The worker does not know which list Alice was assigned. [Serve](/articles/the-same-customer-keeps-the-same-list/) does. If any row has `variant`, stop.

## The worker

One consumer group owns the SQL table. The group id is yours. Cicerone does not set one. A second group writing the same table will race. Members of this group are fine: a user's key stays on one partition.

Commit the offset after the SQL transaction commits. `EnableAutoCommit` stays false. The same `message_id` is a no-op, then you still commit the offset. A body that does not deserialize is poison: park it, then commit. This sample throws and leaves the offset uncommitted.

`Confluent.Kafka` and `Microsoft.Data.SqlClient`. .NET 8.

```csharp
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
    var record = consumer.Consume(cts.Token);
    var message = System.Text.Json.JsonSerializer.Deserialize<RecommendationMessage>(record.Message.Value)
        ?? throw new InvalidOperationException("empty recommendation message");
    if (message.Recommendations.Exists(row => row.Variant is not null))
        throw new InvalidOperationException("variant rows belong to serve assignment");
    await ApplyAsync(sql, message, cts.Token);
    consumer.Commit(record);
}

static async Task ApplyAsync(string sql, RecommendationMessage message, CancellationToken ct)
{
    await using var connection = new SqlConnection(sql);
    await connection.OpenAsync(ct);
    await using var tx = (SqlTransaction)await connection.BeginTransactionAsync(ct);

    await using (var current = new SqlCommand(
        "SELECT message_id FROM cicerone_recommendation_messages WHERE user_id = @user_id",
        connection, tx))
    {
        current.Parameters.AddWithValue("@user_id", message.UserId);
        var applied = (string?)await current.ExecuteScalarAsync(ct);
        if (applied == message.MessageId)
        {
            await tx.CommitAsync(ct);
            return;
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
        insert.Parameters.AddWithValue("@user_id", row.UserId);
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

An empty `recommendations` array takes the delete branch and writes no items. That is the incremental clear. The nightly job does not send that array for a user it dropped.

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

The page:

```sql
SELECT item_id, rank, score, source
FROM cicerone_recommendations
WHERE user_id = @userId
ORDER BY rank;
```

Zero rows means you have never applied a message for that user. Read `__cold_start__`. Rows from last night mean the topic has not said otherwise.

## After the write

Publish runs only when the recommendations write succeeded and this run is still the latest manifest. A newer `generated_at` logs `Skipping publish: recommendations were superseded` (incremental: `Skipping incremental publish: recommendations were superseded`). If the manifest cannot be confirmed, the log is `Skipping publish: could not confirm sidecar generation` and nothing is produced.

Connect, delivery, timeout, and a non-JSON score are logged and left there:

```text
Publish failed after successful write
Incremental publish failed after successful write
```

The job stays successful. The incremental flush is already applied. SQL Server can sit on the previous message until a later publish lands. A lost retrain or apply fence still fails the run. That log line is not the fence.

Operator detail is in [incremental events](/incremental-events/).

## When you should not do this

You share Cicerone's database. Join it. A nightly replace drops the user in the same write. This topic will not.

The storefront can call `GET /recommendations/{user_id}`. That response is the current list. An unknown user comes back empty. You do not keep a second copy, and you do not invent a delete rule.

You turned `[experiment]` on. Assignment is a hash in serve. This JSON is the rows the job wrote, recipes included.

You wanted the topic to be the source of truth. It is a copy, produced after the table, skipped when the generation is stale, and omitted when produce fails.

## In the morning

Alice had twenty rows last night. Tonight's job did not put her in the frame. Kafka said nothing. SQL Server still has the twenty.

Bob's incremental flush cleared him. His message has `"recommendations": []`. Those rows are gone.

Silence keeps yesterday's list. The empty array is the clear, and the nightly publish does not send it for someone who disappeared.
