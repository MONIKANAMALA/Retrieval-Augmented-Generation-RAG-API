"""Embedding service with pluggable backends.

Provides:
- SentenceTransformerEmbedder: real local embeddings via sentence-transformers.
- DeterministicEmbedder: reproducible fake embeddings for offline tests.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

import numpy as np

from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", model_name)
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded. Dimension: %d", self._dimension)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {exc}") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class DeterministicEmbedder(BaseEmbedder):
    """Deterministic embeddings for offline testing.

    Produces reproducible vectors derived from text hashes.
    NOT suitable for semantic search — intended only for testing
    ingestion, retrieval wiring, and API contracts.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        logger.warning(
            "Using DeterministicEmbedder (dim=%d). "
            "This does NOT provide real semantic search.",
            dimension,
        )

    def _hash_to_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit vector from text."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand hash to fill dimension using chained hashes
        expanded = b""
        i = 0
        while len(expanded) < self._dimension * 4:
            expanded += hashlib.sha256(h + i.to_bytes(4, "big")).digest()
            i += 1
        # Use int32 and convert to float to avoid inf/nan from raw float32 bits
        int_arr = np.frombuffer(expanded[: self._dimension * 4], dtype=np.int32).copy()
        arr = int_arr.astype(np.float64) / (2**31)  # Normalize to [-1, 1]
        # Normalize to unit vector
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    @property
    def dimension(self) -> int:
        return self._dimension


# Global embedder instance (set during app startup)
_embedder: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """Return the global embedder instance."""
    global _embedder
    if _embedder is None:
        raise EmbeddingError("Embedder not initialized. Call set_embedder() first.")
    return _embedder


def set_embedder(embedder: BaseEmbedder) -> None:
    """Set the global embedder instance."""
    global _embedder
    _embedder = embedder
