"""Query and health-check API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.exceptions import GenerationConfigError, GenerationError, RetrievalError
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, ReadyResponse
from app.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

# Module-level reference, set during app startup
_pipeline: RAGPipeline | None = None


def set_pipeline(pipeline: RAGPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def _get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")
    return _pipeline


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using the RAG pipeline."""
    pipeline = _get_pipeline()

    try:
        result = pipeline.answer(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
        return result
    except GenerationConfigError:
        raise HTTPException(
            status_code=503,
            detail="LLM provider is not configured. Set MISTRAL_API_KEY in environment.",
        )
    except GenerationError as exc:
        logger.error("Generation error: %s", exc)
        if "rate limit" in str(exc).lower():
            raise HTTPException(
                status_code=429,
                detail="LLM rate limit exceeded. Please check your Mistral API quota or try again later.",
            )
        raise HTTPException(
            status_code=502,
            detail="Answer generation failed. Please try again later.",
        )
    except RetrievalError as exc:
        logger.error("Retrieval error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Retrieval failed. Please try again later.",
        )
    except Exception:
        logger.exception("Unexpected error in query endpoint")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred.",
        )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@router.get("/ready", response_model=ReadyResponse)
async def ready():
    """Readiness check endpoint."""
    settings = get_settings()
    return ReadyResponse(
        status="ready",
        vector_store="chromadb",
        llm_provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL,
    )
