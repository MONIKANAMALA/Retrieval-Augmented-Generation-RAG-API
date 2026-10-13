"""ChromaDB vector store service."""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.services.chunking import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper around ChromaDB for storing and querying document chunks."""

    def __init__(self, settings: Settings | None = None, client: chromadb.ClientAPI | None = None):
        if settings is None:
            from app.core.config import get_settings
            settings = get_settings()

        self._settings = settings

        try:
            if client is not None:
                self._client = client
            else:
                self._client = chromadb.PersistentClient(
                    path=str(settings.chroma_path)
                )
            self._collection = self._client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "l2"},  # L2 distance
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d items).",
                settings.CHROMA_COLLECTION_NAME,
                self._collection.count(),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {exc}") from exc

    @property
    def collection_count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings to the vector store."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number if c.page_number is not None else -1,
            }
            for c in chunks
        ]

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Upserted %d chunks into vector store.", len(chunks))
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert chunks: {exc}") from exc

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store and return matched chunks with metadata.

        Returns a list of dicts with keys:
            chunk_id, document_id, filename, chunk_index, page_number, text, distance
        """
        if self._collection.count() == 0:
            return []

        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": {"$eq": document_ids[0]}}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Query failed: {exc}") from exc

        matched: list[dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return matched

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            page_num = meta.get("page_number")
            if page_num == -1:
                page_num = None

            matched.append({
                "chunk_id": chunk_id,
                "document_id": meta["document_id"],
                "filename": meta["filename"],
                "chunk_index": meta["chunk_index"],
                "page_number": page_num,
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })

        return matched

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a given document_id. Returns count deleted."""
        try:
            # Get all chunks for this document
            existing = self._collection.get(
                where={"document_id": {"$eq": document_id}},
                include=[],
            )
            if not existing or not existing["ids"]:
                return 0

            chunk_ids = existing["ids"]
            self._collection.delete(ids=chunk_ids)
            logger.info("Deleted %d chunks for document '%s'.", len(chunk_ids), document_id)
            return len(chunk_ids)
        except Exception as exc:
            raise VectorStoreError(f"Delete failed: {exc}") from exc

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """Get all chunks for a document."""
        try:
            results = self._collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Get failed: {exc}") from exc

        chunks = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                page_num = meta.get("page_number")
                if page_num == -1:
                    page_num = None
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "chunk_index": meta["chunk_index"],
                    "page_number": page_num,
                    "text": results["documents"][i],
                })
        return sorted(chunks, key=lambda c: c["chunk_index"])

    def list_documents(self) -> list[dict[str, Any]]:
        """List all unique documents in the store."""
        try:
            all_data = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"List failed: {exc}") from exc

        docs: dict[str, dict[str, Any]] = {}
        if all_data and all_data["ids"]:
            for meta in all_data["metadatas"]:
                doc_id = meta["document_id"]
                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": meta["filename"],
                        "chunk_count": 0,
                    }
                docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def has_document(self, document_id: str) -> bool:
        """Check if a document exists in the store."""
        try:
            results = self._collection.get(
                where={"document_id": {"$eq": document_id}},
                include=[],
            )
            return bool(results and results["ids"])
        except Exception:
            return False
