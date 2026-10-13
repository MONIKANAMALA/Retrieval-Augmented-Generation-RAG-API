"""Shared test fixtures and conftest for the test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure deterministic embeddings and mock generator in tests
os.environ["USE_DETERMINISTIC_EMBEDDER"] = "true"
os.environ["MISTRAL_API_KEY"] = ""
os.environ["CHROMA_PERSIST_DIRECTORY"] = ""  # Will be overridden per test
os.environ["APP_ENV"] = "test"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import Settings
from app.services.embeddings import DeterministicEmbedder, set_embedder
from app.services.generation import MockGenerator, set_generator
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore


@pytest.fixture
def settings(tmp_path):
    """Create test settings with a temporary ChromaDB directory."""
    return Settings(
        APP_ENV="test",
        MISTRAL_API_KEY="",
        CHROMA_PERSIST_DIRECTORY=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME="test_collection",
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=30,
        DEFAULT_TOP_K=5,
        MIN_RELEVANCE_SCORE=3.0,  # Lenient for deterministic embeddings (L2 up to 2.0)
    )


@pytest.fixture
def embedder():
    """Create and set a deterministic embedder."""
    emb = DeterministicEmbedder(dimension=384)
    set_embedder(emb)
    return emb


@pytest.fixture
def mock_generator():
    """Create and set a mock generator."""
    gen = MockGenerator()
    set_generator(gen)
    return gen


@pytest.fixture
def vector_store(settings, tmp_path):
    """Create a vector store with a fresh, isolated collection per test."""
    import uuid

    import chromadb
    # Use a unique collection name per test to prevent data leakage
    settings.CHROMA_COLLECTION_NAME = f"test_{uuid.uuid4().hex[:12]}"
    client = chromadb.Client()
    store = VectorStore(settings=settings, client=client)
    return store


@pytest.fixture
def retrieval_service(vector_store, settings, embedder):
    """Create a retrieval service."""
    return RetrievalService(vector_store, settings)


@pytest.fixture
def rag_pipeline(retrieval_service, mock_generator):
    """Create a RAG pipeline with mock generator."""
    return RAGPipeline(retrieval_service)


@pytest.fixture
def sample_texts():
    """Sample text content for testing."""
    return {
        "refund_policy": (
            "All customers are eligible for a full refund within 30 days of purchase. "
            "Refund requests must be submitted through the customer portal or by contacting "
            "customer support. Refunds are processed within 5-7 business days after approval."
        ),
        "remote_work": (
            "Eligible employees may work remotely up to 3 days per week with manager approval. "
            "The company provides a one-time $500 home office setup stipend for approved remote workers. "
            "Remote employees are required to be available during core business hours of 10:00 AM to 3:00 PM."
        ),
        "product_specs": (
            "The AcmeWidget Pro X200 weighs 2.4 kg and has dimensions of 30cm x 20cm x 8cm. "
            "Maximum power consumption is 45 watts. The device supports Wi-Fi 6 and Bluetooth 5.2."
        ),
    }
