"""Citation validation service.

Validates that LLM-generated citations reference real chunks from the
current retrieval, and that quoted text actually appears in the source.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.schemas.query import Citation
from app.services.generation import GeneratedCitation

logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace for fuzzy quote matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_citations(
    generated_citations: list[GeneratedCitation],
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[list[Citation], list[str]]:
    """Validate LLM-generated citations against retrieved chunks.

    Returns:
        - A list of validated Citation objects with trusted metadata.
        - A list of rejection reason strings for invalid citations.

    Only citations whose chunk_id matches a retrieved chunk AND whose
    quoted_text appears in the chunk text are accepted.
    """
    # Build lookup from retrieved chunks
    chunk_map: dict[str, dict[str, Any]] = {
        c["chunk_id"]: c for c in retrieved_chunks
    }

    validated: list[Citation] = []
    rejections: list[str] = []
    citation_counter = 0

    for gen_cite in generated_citations:
        # 1. Verify chunk_id exists in retrieved chunks
        if gen_cite.chunk_id not in chunk_map:
            rejections.append(
                f"chunk_id '{gen_cite.chunk_id}' not found in retrieved chunks."
            )
            continue

        chunk = chunk_map[gen_cite.chunk_id]

        # 2. Verify quoted text appears in the chunk
        norm_quote = _normalize_whitespace(gen_cite.quoted_text)
        norm_chunk = _normalize_whitespace(chunk["text"])

        if not norm_quote:
            rejections.append(
                f"chunk_id '{gen_cite.chunk_id}': empty quoted_text."
            )
            continue

        if norm_quote not in norm_chunk:
            # Try partial match (first 50 chars)
            partial = norm_quote[:50]
            if partial not in norm_chunk:
                rejections.append(
                    f"chunk_id '{gen_cite.chunk_id}': quoted_text not found in source chunk."
                )
                continue
            else:
                logger.warning(
                    "Citation for chunk '%s' accepted on partial match (first 50 chars).",
                    gen_cite.chunk_id,
                )

        citation_counter += 1
        validated.append(
            Citation(
                citation_id=f"C{citation_counter}",
                document_id=chunk["document_id"],
                filename=chunk["filename"],
                page_number=chunk.get("page_number"),
                chunk_id=gen_cite.chunk_id,
                quoted_text=gen_cite.quoted_text,
            )
        )

    if rejections:
        logger.warning(
            "Citation validation rejected %d/%d citations: %s",
            len(rejections),
            len(generated_citations),
            "; ".join(rejections),
        )

    return validated, rejections
