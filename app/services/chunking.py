"""Text chunking service with configurable size and overlap."""

from __future__ import annotations

import hashlib
import logging
from typing import NamedTuple

from app.services.document_parser import ParsedPage

logger = logging.getLogger(__name__)


class Chunk(NamedTuple):
    """A single text chunk with metadata."""
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    text: str


def _generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a deterministic, unique chunk ID."""
    raw = f"{document_id}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """Split text into overlapping chunks.

    Tries to break on sentence boundaries ('. ') when possible.
    Returns a list of non-empty chunk strings.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at sentence boundary
        if end < text_len:
            # Look for '. ' within the last 20% of the chunk
            search_start = max(start, end - chunk_size // 5)
            last_period = text.rfind(". ", search_start, end)
            if last_period > start:
                end = last_period + 2  # Include the period and space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - chunk_overlap

    return chunks


def chunk_pages(
    pages: list[ParsedPage],
    document_id: str,
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk all pages of a parsed document.

    Each page is chunked independently to preserve page number metadata.
    Returns a flat list of Chunk objects with sequential indices.
    """
    all_chunks: list[Chunk] = []
    global_index = 0

    for page in pages:
        text_chunks = chunk_text(page.text, chunk_size, chunk_overlap)
        for text in text_chunks:
            chunk_id = _generate_chunk_id(document_id, global_index)
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=global_index,
                    page_number=page.page_number,
                    text=text,
                )
            )
            global_index += 1

    logger.info(
        "Chunked document '%s' (%s) into %d chunks",
        filename, document_id, len(all_chunks),
    )
    return all_chunks
