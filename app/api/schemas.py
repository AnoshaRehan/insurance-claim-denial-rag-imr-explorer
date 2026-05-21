from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask.")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Number of records to retrieve."
    )
    filter_diagnosis: str | None = Field(
        default=None, description="Optional diagnosis category filter."
    )


class Source(BaseModel):
    """A single source record cited in an answer."""

    reference_id: str | None
    diagnosis_category: str | None
    determination: str | None
    score: float


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    question: str
    answer: str
    sources: list[Source]
    low_confidence: bool


class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""

    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_diagnosis: str | None = None


class SearchResult(BaseModel):
    """A single search result."""

    reference_id: str | None
    diagnosis_category: str | None
    diagnosis_subcategory: str | None
    determination: str | None
    findings: str
    score: float


class SearchResponse(BaseModel):
    """Response body for the /search endpoint."""

    query: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    environment: str
    qdrant_connected: bool
    indexed_points: int | None = None
