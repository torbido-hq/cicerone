"""Pydantic models for the serve API — drive FastAPI's OpenAPI schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class RecommendationItem(BaseModel):
    item_id: str = Field(description="Catalog item identifier")
    rank: int = Field(ge=1, description="1-based position in the returned list")
    score: float = Field(description="Score from the batch job (strategy- or blend-specific)")
    source: str = Field(
        description="Origin label, e.g. personalized, item_based, content_fallback, popular_fallback, latest, blended",
        examples=["blended"],
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
