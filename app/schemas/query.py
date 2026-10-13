"""Pydantic schemas for query requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve.")
    document_ids: list[str] | None = Field(default=None, description="Optional filter by document IDs.")


class Citation(BaseModel):
    """A single citation reference."""
    citation_id: str
    document_id: str
    filename: str
    page_number: int | None = None
    chunk_id: str
    quoted_text: str


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    answer: str
    abstained: bool
    abstention_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: int
    request_id: str


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    """Response for readiness check."""
    status: str
    vector_store: str
    llm_provider: str
    embedding_model: str
