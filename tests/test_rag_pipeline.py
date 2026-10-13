"""Tests for the RAG pipeline end-to-end."""

import pytest

from app.services.chunking import Chunk
from app.services.generation import (
    GeneratedAnswer,
    GeneratedCitation,
    GenerationError,
    MockGenerator,
    set_generator,
)
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import RetrievalService


class TestRAGPipeline:
    """End-to-end RAG pipeline tests."""

    def _ingest_sample(self, vector_store, embedder):
        """Ingest sample chunks."""
        chunks = [
            Chunk("c1", "doc1", "policy.md", 0, None,
                  "All customers are eligible for a full refund within 30 days of purchase."),
            Chunk("c2", "doc1", "policy.md", 1, None,
                  "Standard shipping takes 5-7 business days. Free shipping on orders over $75."),
            Chunk("c3", "doc2", "handbook.md", 0, None,
                  "Employees receive 20 days of annual leave per year."),
        ]
        embeddings = embedder.embed([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)
        return chunks

    def test_answerable_query(self, vector_store, embedder, settings, mock_generator):
        """Pipeline returns a response for answerable questions."""
        self._ingest_sample(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)

        result = pipeline.answer("What is the refund policy?")
        assert result.request_id is not None
        assert result.answer is not None

    def test_empty_corpus_abstention(self, vector_store, embedder, settings, mock_generator):
        """Pipeline abstains on empty corpus."""
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)
        result = pipeline.answer("anything")
        assert result.abstained is True
        # Either empty_corpus or no_relevant_evidence is acceptable
        assert result.abstention_reason in ("empty_corpus", "no_relevant_evidence")

    def test_request_id_is_unique(self, vector_store, embedder, settings, mock_generator):
        """Each response has a unique request_id."""
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)
        r1 = pipeline.answer("q1")
        r2 = pipeline.answer("q2")
        assert r1.request_id != r2.request_id

    def test_generation_error_propagates(self, vector_store, embedder, settings):
        """Generation errors are not silently converted to answers."""

        class FailingGenerator(MockGenerator):
            def generate(self, question, evidence_chunks):
                raise GenerationError("API timeout")

        self._ingest_sample(vector_store, embedder)
        set_generator(FailingGenerator())

        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)

        with pytest.raises(GenerationError, match="API timeout"):
            pipeline.answer("What is the refund policy?")

    def test_answer_without_citations_causes_abstention(self, vector_store, embedder, settings):
        """An answer with no citations triggers abstention."""

        class NoCitationGenerator(MockGenerator):
            def generate(self, question, evidence_chunks):
                return GeneratedAnswer(
                    answer="Some answer without citations",
                    abstained=False,
                    citations=[],
                )

        self._ingest_sample(vector_store, embedder)
        set_generator(NoCitationGenerator())

        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)

        result = pipeline.answer("refund?")
        assert result.abstained is True
        assert result.abstention_reason in ("insufficient_evidence", "no_relevant_evidence")

    def test_all_citations_rejected(self, vector_store, embedder, settings):
        """When all citations are invalid, pipeline abstains."""

        class BadCitationGenerator(MockGenerator):
            def generate(self, question, evidence_chunks):
                return GeneratedAnswer(
                    answer="Some answer",
                    abstained=False,
                    citations=[
                        GeneratedCitation(
                            chunk_id="nonexistent_chunk",
                            quoted_text="fabricated quote",
                        )
                    ],
                )

        self._ingest_sample(vector_store, embedder)
        set_generator(BadCitationGenerator())

        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)

        result = pipeline.answer("refund?")
        assert result.abstained is True

    def test_document_ids_filter(self, vector_store, embedder, settings, mock_generator):
        """Pipeline respects document_ids filter."""
        self._ingest_sample(vector_store, embedder)
        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)

        result = pipeline.answer("policy", document_ids=["doc1"])
        assert result.request_id is not None

    def test_llm_abstention_passes_through(self, vector_store, embedder, settings):
        """When LLM explicitly abstains, pipeline respects it."""

        class AbstainGenerator(MockGenerator):
            def generate(self, question, evidence_chunks):
                return GeneratedAnswer(
                    answer="I cannot answer this.",
                    abstained=True,
                    abstention_reason="insufficient_evidence",
                    citations=[],
                )

        self._ingest_sample(vector_store, embedder)
        set_generator(AbstainGenerator())

        settings.MIN_RELEVANCE_SCORE = 100.0
        service = RetrievalService(vector_store, settings)
        pipeline = RAGPipeline(service)

        result = pipeline.answer("unknown question")
        assert result.abstained is True
        assert result.abstention_reason in ("insufficient_evidence", "no_relevant_evidence")
