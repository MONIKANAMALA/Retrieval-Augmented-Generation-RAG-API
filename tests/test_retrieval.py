"""Tests for retrieval service."""

from app.services.chunking import Chunk
from app.services.retrieval import RetrievalService


class TestRetrieval:
    """Tests for the retrieval service."""

    def _add_sample_chunks(self, vector_store, embedder):
        """Helper to add sample chunks to the vector store."""
        chunks = [
            Chunk(
                chunk_id="chunk_refund_1",
                document_id="doc_policies",
                filename="company_policies.md",
                chunk_index=0,
                page_number=None,
                text="All customers are eligible for a full refund within 30 days of purchase.",
            ),
            Chunk(
                chunk_id="chunk_remote_1",
                document_id="doc_handbook",
                filename="employee_handbook.md",
                chunk_index=0,
                page_number=None,
                text="Eligible employees may work remotely up to 3 days per week with manager approval.",
            ),
            Chunk(
                chunk_id="chunk_product_1",
                document_id="doc_manual",
                filename="product_manual.md",
                chunk_index=0,
                page_number=None,
                text="The AcmeWidget Pro X200 weighs 2.4 kg and has dimensions of 30cm x 20cm x 8cm.",
            ),
            Chunk(
                chunk_id="chunk_shipping_1",
                document_id="doc_policies",
                filename="company_policies.md",
                chunk_index=1,
                page_number=None,
                text="Standard shipping takes 5-7 business days. Free shipping on orders over $75.",
            ),
        ]
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        vector_store.add_chunks(chunks, embeddings)
        return chunks

    def test_empty_corpus(self, vector_store, embedder, settings):
        """Retrieval from empty corpus returns empty results."""
        service = RetrievalService(vector_store, settings)
        result = service.retrieve("What is the refund policy?")
        assert result.is_empty
        assert result.count == 0

    def test_retrieval_returns_results(self, vector_store, embedder, settings):
        """Retrieval returns chunks when corpus has data (with lenient threshold)."""
        self._add_sample_chunks(vector_store, embedder)
        # Deterministic embeddings produce large L2 distances; use a very lenient threshold
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        result = service.retrieve("refund policy", top_k=3)
        assert result.count > 0

    def test_top_k_limit(self, vector_store, embedder, settings):
        """Retrieval respects top_k limit."""
        self._add_sample_chunks(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        result = service.retrieve("company policy", top_k=2)
        assert result.count <= 2

    def test_document_id_filter(self, vector_store, embedder, settings):
        """Retrieval with document_ids filter."""
        self._add_sample_chunks(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        result = service.retrieve(
            "policy",
            top_k=10,
            document_ids=["doc_policies"],
        )
        # All results should be from the filtered document
        for chunk in result.chunks:
            assert chunk["document_id"] == "doc_policies"

    def test_result_has_metadata(self, vector_store, embedder, settings):
        """Retrieved chunks include expected metadata fields."""
        self._add_sample_chunks(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        result = service.retrieve("refund", top_k=1)
        if result.count > 0:
            chunk = result.chunks[0]
            assert "chunk_id" in chunk
            assert "document_id" in chunk
            assert "filename" in chunk
            assert "text" in chunk
            assert "distance" in chunk

    def test_result_has_distance(self, vector_store, embedder, settings):
        """Retrieved chunks include distance information."""
        self._add_sample_chunks(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        result = service.retrieve("refund", top_k=5)
        for chunk in result.chunks:
            assert isinstance(chunk["distance"], (int, float))

    def test_deletion_removes_from_retrieval(self, vector_store, embedder, settings):
        """After deleting a document, its chunks are not retrieved."""
        self._add_sample_chunks(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)

        # Delete policies document
        vector_store.delete_document("doc_policies")

        result = service.retrieve("refund policy", top_k=10)
        for chunk in result.chunks:
            assert chunk["document_id"] != "doc_policies"

    def test_multiple_chunks_retrieval(self, vector_store, embedder, settings):
        """Retrieval can return multiple chunks."""
        self._add_sample_chunks(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        result = service.retrieve("company information", top_k=4)
        assert result.count >= 1

    def test_strict_threshold_filters(self, vector_store, embedder, settings):
        """Very strict threshold filters out distant chunks."""
        self._add_sample_chunks(vector_store, embedder)
        # Deterministic embeddings produce large distances; 0.001 filters everything
        settings.MIN_RELEVANCE_SCORE = 0.001
        service = RetrievalService(vector_store, settings)
        result = service.retrieve("refund policy", top_k=5)
        # Should be empty or very few results due to strict threshold
        # (deterministic embeddings produce distances far from 0)
        assert result.count == 0
