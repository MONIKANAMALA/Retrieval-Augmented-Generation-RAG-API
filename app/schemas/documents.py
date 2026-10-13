"""Pydantic schemas for document-related requests and responses."""

from __future__ import annotations

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata for a stored document."""
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    total_characters: int
    pages: int | None = None


class IngestionResponse(BaseModel):
    """Response after document ingestion."""
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    total_characters: int
    pages: int | None = None
    status: str = "ingested"
    message: str = "Document successfully ingested."


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: list[DocumentMetadata]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""
    document_id: str
    filename: str
    deleted_chunks: int
    status: str = "deleted"


class ChunkInfo(BaseModel):
    """Information about a single chunk."""
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None = None
    text: str
    character_count: int
