"""Retrieval service: embeds queries and retrieves relevant chunks."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.services.embeddings import get_embedder
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalResult:
    """Container for retrieval results."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        top_k: int,
        relevance_threshold: float | None = None,
    ):
        self.chunks = chunks
        self.query = query
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


class RetrievalService:
    """Semantic retrieval using embeddings and vector store."""

    def __init__(self, vector_store: VectorStore, settings: Settings | None = None):
        self._store = vector_store
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        ChromaDB uses L2 (Euclidean) distance, where lower = more similar.
        Chunks with distance > MIN_RELEVANCE_SCORE are filtered out.
        """
        if top_k is None:
            top_k = self._settings.DEFAULT_TOP_K

        # Check for empty collection
        if self._store.collection_count == 0:
            logger.info("Empty corpus — returning no results.")
            return RetrievalResult(
                chunks=[], query=query, top_k=top_k,
                relevance_threshold=self._settings.MIN_RELEVANCE_SCORE,
            )

        try:
            embedder = get_embedder()
            query_embedding = embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc

        try:
            raw_results = self._store.query(
                query_embedding=query_embedding,
                top_k=top_k,
                document_ids=document_ids,
            )
        except Exception as exc:
            raise RetrievalError(f"Vector store query failed: {exc}") from exc

        # Filter by relevance threshold (L2 distance: lower is better)
        threshold = self._settings.MIN_RELEVANCE_SCORE
        filtered = []
        for chunk in raw_results:
            distance = chunk.get("distance", float("inf"))
            if distance <= threshold:
                filtered.append(chunk)
            else:
                logger.debug(
                    "Chunk %s filtered out (distance=%.4f > threshold=%.4f)",
                    chunk["chunk_id"], distance, threshold,
                )

        logger.info(
            "Retrieved %d/%d chunks for query (threshold=%.2f): '%s'",
            len(filtered), len(raw_results), threshold, query[:80],
        )

        return RetrievalResult(
            chunks=filtered,
            query=query,
            top_k=top_k,
            relevance_threshold=threshold,
        )
