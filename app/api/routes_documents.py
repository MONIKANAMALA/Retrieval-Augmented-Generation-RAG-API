"""Document management API routes."""

from __future__ import annotations

import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.schemas.documents import (
    ChunkInfo,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    IngestionResponse,
)
from app.services.chunking import chunk_pages
from app.services.document_parser import parse_document, validate_extension
from app.services.embeddings import get_embedder
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Module-level reference to the shared vector store, set during app startup
_vector_store: VectorStore | None = None


def set_vector_store(store: VectorStore) -> None:
    global _vector_store
    _vector_store = store


def _get_store() -> VectorStore:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized.")
    return _vector_store


def _generate_document_id(filename: str, content: bytes) -> str:
    """Deterministic document ID from filename + content hash."""
    raw = f"{filename}::{hashlib.sha256(content).hexdigest()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


@router.post("/upload", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: Annotated[UploadFile, File(description="Document file to upload")]):
    """Upload and ingest a document (PDF, TXT, or Markdown)."""
    settings = get_settings()
    store = _get_store()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    # Validate extension first
    try:
        ext = validate_extension(file.filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    # Read content
    content = await file.read()

    # Validate size
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Parse document
    try:
        parsed = parse_document(content, file.filename)
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DocumentParsingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Generate document ID (deterministic: same file → same ID → upsert replaces)
    doc_id = _generate_document_id(file.filename, content)

    # If document already exists, delete old chunks first (deterministic replacement)
    if store.has_document(doc_id):
        store.delete_document(doc_id)
        logger.info("Replaced existing document '%s' (id=%s).", file.filename, doc_id)

    # Chunk
    chunks = chunk_pages(
        pages=parsed.pages,
        document_id=doc_id,
        filename=file.filename,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from document.")

    # Embed
    try:
        embedder = get_embedder()
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    # Store
    store.add_chunks(chunks, embeddings)

    return IngestionResponse(
        document_id=doc_id,
        filename=file.filename,
        file_type=ext,
        chunk_count=len(chunks),
        total_characters=parsed.total_characters,
        pages=parsed.page_count,
        status="ingested",
        message=f"Document '{file.filename}' ingested successfully with {len(chunks)} chunks.",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents."""
    store = _get_store()
    docs = store.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentMetadata(
                document_id=d["document_id"],
                filename=d["filename"],
                file_type="",
                chunk_count=d["chunk_count"],
                total_characters=0,
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=list[ChunkInfo])
async def get_document(document_id: str):
    """Get all chunks for a specific document."""
    store = _get_store()
    chunks = store.get_document_chunks(document_id)

    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    return [
        ChunkInfo(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            page_number=c.get("page_number"),
            text=c["text"],
            character_count=len(c["text"]),
        )
        for c in chunks
    ]


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    store = _get_store()

    if not store.has_document(document_id):
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    # Get filename before deletion
    chunks = store.get_document_chunks(document_id)
    filename = chunks[0]["filename"] if chunks else "unknown"
    deleted = store.delete_document(document_id)

    return DocumentDeleteResponse(
        document_id=document_id,
        filename=filename,
        deleted_chunks=deleted,
        status="deleted",
    )
