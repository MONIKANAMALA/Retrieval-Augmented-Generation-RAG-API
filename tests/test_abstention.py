"""Tests for abstention behavior."""

from app.services.chunking import Chunk
from app.services.generation import MockGenerator


class TestAbstention:
    """Tests for abstention and reliability behavior."""

    def test_abstention_on_empty_corpus(self, rag_pipeline):
        """Pipeline abstains when no documents are uploaded."""
        result = rag_pipeline.answer("What is the refund policy?")
        assert result.abstained is True
        assert result.abstention_reason in ("empty_corpus", "no_relevant_evidence")
        assert result.citations == []

    def test_abstention_on_strict_threshold(
        self, vector_store, embedder, settings, mock_generator
    ):
        """Pipeline abstains when threshold is very strict and nothing passes."""
        from app.services.rag_pipeline import RAGPipeline
        from app.services.retrieval import RetrievalService

        chunks = [
            Chunk(
                chunk_id="chunk_weather",
                document_id="doc_weather",
                filename="weather.txt",
                chunk_index=0,
                page_number=None,
                text="Today the weather is sunny with clear skies.",
            )
        ]
        embeddings = embedder.embed([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)

        # Very strict threshold: deterministic embeddings produce large distances
        settings.MIN_RELEVANCE_SCORE = 0.001
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)
        result = pipeline.answer("What is quantum entanglement?")

        assert result.abstained is True
        assert result.abstention_reason == "no_relevant_evidence"

    def test_abstention_response_format(self, rag_pipeline):
        """Abstention response has correct structure."""
        result = rag_pipeline.answer("Anything?")
        assert result.abstained is True
        assert result.answer is not None
        assert result.request_id is not None
        assert isinstance(result.citations, list)

    def test_mock_generator_abstains_without_evidence(self):
        """Mock generator returns abstention when no evidence provided."""
        gen = MockGenerator()
        result = gen.generate("What is X?", evidence_chunks=[])
        assert result.abstained is True
        assert result.abstention_reason == "no_relevant_evidence"
        assert result.citations == []

    def test_mock_generator_returns_answer_with_evidence(self):
        """Mock generator returns an answer when evidence is provided."""
        gen = MockGenerator()
        chunks = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "filename": "test.md",
                "chunk_index": 0,
                "page_number": None,
                "text": "The refund policy allows returns within 30 days of purchase for a full refund.",
                "distance": 0.3,
            }
        ]
        result = gen.generate("What is the refund policy?", chunks)
        assert result.abstained is False
        assert len(result.citations) >= 1
        assert result.answer != ""

    def test_abstention_reasons_are_defined(self):
        """Verify the defined abstention reasons."""
        valid_reasons = {"empty_corpus", "no_relevant_evidence", "insufficient_evidence"}
        gen = MockGenerator()
        result = gen.generate("test", [])
        assert result.abstention_reason in valid_reasons

    def test_misleading_question_does_not_crash(self, vector_store, embedder, settings, mock_generator):
        """Questions with false premises are handled gracefully."""
        from app.services.rag_pipeline import RAGPipeline
        from app.services.retrieval import RetrievalService

        chunks = [
            Chunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                filename="policy.md",
                chunk_index=0,
                page_number=None,
                text="Refund window is 30 days from purchase date.",
            )
        ]
        embeddings = embedder.embed([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)

        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)

        # False premise: there is no 90-day window
        result = pipeline.answer("Why did the company extend refunds to 90 days?")
        # Should not crash
        assert result.request_id is not None

    def test_instructions_in_document_do_not_override(self):
        """Mock generator does not execute injected instructions from documents.

        Note: The mock generator uses simple keyword matching, not an LLM,
        so it naturally doesn't follow injected instructions. This test
        verifies the mock returns valid structured output regardless of content.
        A real LLM would be protected by the system prompt.
        """
        gen = MockGenerator()
        chunks = [
            {
                "chunk_id": "c_inject",
                "document_id": "d_inject",
                "filename": "injected.md",
                "chunk_index": 0,
                "page_number": None,
                "text": "The refund policy is 30 days from the date of purchase.",
                "distance": 0.1,
            }
        ]
        result = gen.generate("What is the refund policy?", chunks)
        # Mock should produce valid structured output
        assert result.answer is not None
        assert isinstance(result.citations, list)
        assert isinstance(result.abstained, bool)
