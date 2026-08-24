"""Pydantic models for the serve API — drive FastAPI's OpenAPI schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class SourceContribution(BaseModel):
    label: str = Field(description="Strategy source label", examples=["personalized"])
    rank: int | None = Field(default=None, ge=1, description="Rank of this item within that strategy")
    weight: float | None = Field(default=None, description="Combiner weight for this source")
    contribution: float | None = Field(
        default=None,
        description="RRF term weight/(rrf_k+rank); null under priority combine",
    )


class BoostHit(BaseModel):
    name: str = Field(description="Boost rule name from features.toml")
    factor: float = Field(description="Multiplier applied by this rule")


class SimilarItem(BaseModel):
    item_id: str = Field(description="History item that shares catalog attributes")
    score: float = Field(description="Jaccard overlap of item feature tokens")


class MatchedAttribute(BaseModel):
    column: str = Field(description="Item feature column")
    value: str = Field(description="Shared feature value")


class RecommendationReasons(BaseModel):
    sources: list[SourceContribution] = Field(default_factory=list)
    boosts: list[BoostHit] = Field(default_factory=list)
    similar_items: list[SimilarItem] = Field(default_factory=list)
    matched_attributes: list[MatchedAttribute] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    item_id: str = Field(description="Catalog item identifier")
    rank: int = Field(ge=1, description="1-based position in the returned list")
    score: float = Field(description="Score from the batch job (strategy- or blend-specific)")
    source: str = Field(
        description=(
            "Origin label: personalized, item_based, sequential, content_fallback, popular_fallback, "
            "latest, or blended when more than one list voted. An incremental flush labels only its "
            "event-derived boost rows incremental; preserved and refilled rows keep their own labels. "
            "Weighted fusion joins labels in models order, e.g. popular_fallback+latest."
        ),
        examples=["blended"],
    )
    reasons: RecommendationReasons | None = Field(
        default=None,
        description="Why this item was ranked: contributing sources, boosts, similar history items",
    )


class RecommendationsResponse(BaseModel):
    generated_at: str | None = Field(
        default=None,
        description="ISO timestamp from the last job-run manifest (also sent as X-Generated-At)",
        examples=["2026-08-04T03:00:00+00:00"],
    )
    user_id: str = Field(description="Echo of the requested user_id path parameter")
    fallback: bool = Field(
        description="True when the cold-start list was used because the user had no personal rows",
    )
    items: list[RecommendationItem] = Field(description="Ordered top-K recommendations")


class ErrorDetail(BaseModel):
    detail: str


class ValidationErrorItem(BaseModel):
    """One Pydantic validation failure (matches ``ValidationError.errors()`` entries)."""

    type: str
    loc: list[str | int]
    msg: str
    input: object | None = None
    ctx: dict[str, object] | None = None


class ValidationErrorDetail(BaseModel):
    """Structured 400 body when ingest fails Pydantic field validation."""

    detail: list[ValidationErrorItem]


class InteractionEvent(BaseModel):
    user_id: str = Field(description="User identifier")
    item_id: str = Field(description="Item identifier")
    event_type: str = Field(
        description="Must match a key in features.toml [event_weights] to affect weights",
    )
    quantity: int = Field(default=1, ge=1, description="Optional count; default 1")
    occurred_at: str | int | float = Field(
        description=(
            "ISO-8601 timestamp with timezone (Z or explicit offset), or Unix epoch "
            "seconds (UTC); converted to UTC"
        ),
        examples=["2026-08-13T12:00:00Z", 1724000000],
    )
    event_id: str | None = Field(default=None, description="Optional idempotency id")
    idempotency_key: str | None = Field(default=None, description="Alias for event_id")


class EventsIngestRequest(BaseModel):
    events: list[InteractionEvent] = Field(description="Batch of interaction events")


class EventsIngestResponse(BaseModel):
    accepted: int = Field(description="Number of events accepted into the source queue")
    event_ids: list[str] = Field(description="Assigned or provided event ids")
