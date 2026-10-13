"""RAG pipeline: orchestrates retrieval → generation → citation validation."""

from __future__ import annotations

import logging
import uuid

from app.core.exceptions import GenerationError, RetrievalError
from app.schemas.query import QueryResponse
from app.services.citations import validate_citations
from app.services.generation import get_generator
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self, retrieval_service: RetrievalService):
        self._retrieval = retrieval_service

    def answer(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> QueryResponse:
        """Run the full RAG pipeline: retrieve → generate → validate → respond."""
        request_id = str(uuid.uuid4())

        # 1. Retrieve relevant chunks
        try:
            retrieval_result = self._retrieval.retrieve(
                query=question,
                top_k=top_k,
                document_ids=document_ids,
            )
        except RetrievalError as exc:
            logger.error("Retrieval failed: %s", exc)
            raise

        # 2. Handle empty corpus
        if self._retrieval._store.collection_count == 0:
            return QueryResponse(
                answer="The requested information is not available in the uploaded documents.",
                abstained=True,
                abstention_reason="empty_corpus",
                citations=[],
                retrieved_chunks=0,
                request_id=request_id,
            )

        # 3. Handle no relevant evidence
        if retrieval_result.is_empty:
            return QueryResponse(
                answer="The requested information is not available in the uploaded documents.",
                abstained=True,
                abstention_reason="no_relevant_evidence",
                citations=[],
                retrieved_chunks=0,
                request_id=request_id,
            )

        # 4. Generate answer
        generator = get_generator()
        try:
            generated = generator.generate(
                question=question,
                evidence_chunks=retrieval_result.chunks,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Unexpected generation error: {exc}") from exc

        # 5. If LLM abstained, return abstention
        if generated.abstained:
            return QueryResponse(
                answer=generated.answer or "The requested information is not available in the uploaded documents.",
                abstained=True,
                abstention_reason=generated.abstention_reason or "insufficient_evidence",
                citations=[],
                retrieved_chunks=retrieval_result.count,
                request_id=request_id,
            )

        # 6. Validate citations
        validated_citations, _rejections = validate_citations(
            generated_citations=generated.citations,
            retrieved_chunks=retrieval_result.chunks,
        )

        # 7. If no valid citations remain, abstain
        if not validated_citations and generated.citations:
            logger.warning(
                "All citations rejected for request %s. Abstaining.",
                request_id,
            )
            return QueryResponse(
                answer="The requested information is not available in the uploaded documents.",
                abstained=True,
                abstention_reason="insufficient_evidence",
                citations=[],
                retrieved_chunks=retrieval_result.count,
                request_id=request_id,
            )

        # 8. If LLM provided answer but no citations at all, abstain
        if not validated_citations and not generated.citations:
            return QueryResponse(
                answer="The requested information is not available in the uploaded documents.",
                abstained=True,
                abstention_reason="insufficient_evidence",
                citations=[],
                retrieved_chunks=retrieval_result.count,
                request_id=request_id,
            )

        return QueryResponse(
            answer=generated.answer,
            abstained=False,
            abstention_reason=None,
            citations=validated_citations,
            retrieved_chunks=retrieval_result.count,
            request_id=request_id,
        )
