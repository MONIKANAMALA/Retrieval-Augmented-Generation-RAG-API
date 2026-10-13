"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes_documents import router as documents_router
from app.api.routes_documents import set_vector_store
from app.api.routes_query import router as query_router
from app.api.routes_query import set_pipeline
from app.core.config import get_settings
from app.services.embeddings import (
    DeterministicEmbedder,
    SentenceTransformerEmbedder,
    set_embedder,
)
from app.services.generation import (
    HuggingFaceGenerator,
    MistralGenerator,
    MockGenerator,
    set_generator,
)
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting RAG Assessment API (env=%s)", settings.APP_ENV)

    # Ensure storage directories exist
    Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Initialize embedder
    use_deterministic = os.environ.get("USE_DETERMINISTIC_EMBEDDER", "").lower() == "true"
    if use_deterministic:
        embedder = DeterministicEmbedder()
    else:
        try:
            embedder = SentenceTransformerEmbedder(settings.EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning(
                "Failed to load sentence-transformers model, falling back to deterministic: %s", exc
            )
            embedder = DeterministicEmbedder()
    set_embedder(embedder)

    # Initialize vector store
    store = VectorStore(settings)
    set_vector_store(store)

    # Initialize generator
    if settings.LLM_PROVIDER == "huggingface":
        try:
            generator = HuggingFaceGenerator(model_name=settings.HF_MODEL_NAME, settings=settings)
        except Exception as exc:
            logger.warning("HuggingFace generator failed, falling back to mock: %s", exc)
            generator = MockGenerator()
    elif settings.LLM_PROVIDER == "mistral" and settings.has_mistral_key:
        try:
            generator = MistralGenerator(settings)
        except Exception as exc:
            logger.warning("Mistral init failed, using mock: %s", exc)
            generator = MockGenerator()
    else:
        logger.info(
            "Using MockGenerator (provider=%s, key_configured=%s).",
            settings.LLM_PROVIDER,
            settings.has_mistral_key,
        )
        generator = MockGenerator()
    set_generator(generator)

    # Initialize RAG pipeline
    retrieval_service = RetrievalService(store, settings)
    pipeline = RAGPipeline(retrieval_service)
    set_pipeline(pipeline)

    logger.info("RAG Assessment API ready.")
    yield
    logger.info("Shutting down RAG Assessment API.")


app = FastAPI(
    title="RAG Assessment API",
    description=(
        "A Retrieval-Augmented Generation API that answers questions "
        "grounded in uploaded documents using semantic search and Mistral."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(query_router)
