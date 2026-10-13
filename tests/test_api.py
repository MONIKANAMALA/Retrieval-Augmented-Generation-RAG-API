"""Tests for the FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes_documents import set_vector_store
from app.api.routes_query import set_pipeline
from app.core.config import Settings
from app.services.embeddings import DeterministicEmbedder, set_embedder
from app.services.generation import MockGenerator, set_generator
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore


@pytest.fixture
def app_client(tmp_path):
    """Create a test client with manually wired dependencies (no lifespan needed)."""
    import uuid

    import chromadb

    # Setup components directly instead of relying on lifespan
    settings = Settings(
        APP_ENV="test",
        MISTRAL_API_KEY="",
        CHROMA_PERSIST_DIRECTORY=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME=f"test_api_{uuid.uuid4().hex[:12]}",
        MIN_RELEVANCE_SCORE=50.0,  # Lenient for deterministic embeddings
        MAX_UPLOAD_SIZE_MB=10,
    )

    embedder = DeterministicEmbedder(dimension=384)
    set_embedder(embedder)

    generator = MockGenerator()
    set_generator(generator)

    client = chromadb.Client()
    store = VectorStore(settings=settings, client=client)
    set_vector_store(store)

    retrieval_service = RetrievalService(store, settings)
    pipeline = RAGPipeline(retrieval_service)
    set_pipeline(pipeline)

    from app.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(app_client):
    """GET /health returns healthy status."""
    async with app_client as client:
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.asyncio
async def test_ready(app_client):
    """GET /ready returns ready status."""
    async with app_client as client:
        r = await client.get("/ready")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert "vector_store" in data
        assert "llm_provider" in data


@pytest.mark.asyncio
async def test_upload_txt(app_client):
    """POST /documents/upload with a valid TXT file."""
    async with app_client as client:
        content = b"This is a test document with enough content to be chunked properly for ingestion."
        r = await client.post(
            "/documents/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] >= 1
        assert data["status"] == "ingested"
        assert "document_id" in data


@pytest.mark.asyncio
async def test_upload_markdown(app_client):
    """POST /documents/upload with a valid Markdown file."""
    async with app_client as client:
        content = b"# Title\n\nSome content here for testing the markdown ingestion pipeline."
        r = await client.post(
            "/documents/upload",
            files={"file": ("test.md", content, "text/plain")},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["filename"] == "test.md"


@pytest.mark.asyncio
async def test_upload_unsupported(app_client):
    """POST /documents/upload with unsupported file type returns 415."""
    async with app_client as client:
        r = await client.post(
            "/documents/upload",
            files={"file": ("test.docx", b"content", "application/octet-stream")},
        )
        assert r.status_code == 415


@pytest.mark.asyncio
async def test_upload_empty_file(app_client):
    """POST /documents/upload with empty file returns 400."""
    async with app_client as client:
        r = await client.post(
            "/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_upload_oversized(app_client):
    """POST /documents/upload with oversized file returns 413."""
    async with app_client as client:
        # Default MAX_UPLOAD_SIZE_MB=10, so 11MB should fail
        big_content = b"x" * (11 * 1024 * 1024)
        r = await client.post(
            "/documents/upload",
            files={"file": ("big.txt", big_content, "text/plain")},
        )
        assert r.status_code == 413


@pytest.mark.asyncio
async def test_list_documents(app_client):
    """GET /documents returns a list."""
    async with app_client as client:
        r = await client.get("/documents")
        assert r.status_code == 200
        data = r.json()
        assert "documents" in data
        assert "total" in data


@pytest.mark.asyncio
async def test_get_document_not_found(app_client):
    """GET /documents/{id} returns 404 for non-existent document."""
    async with app_client as client:
        r = await client.get("/documents/nonexistent123")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_not_found(app_client):
    """DELETE /documents/{id} returns 404 for non-existent document."""
    async with app_client as client:
        r = await client.delete("/documents/nonexistent123")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_query_empty_corpus(app_client):
    """POST /query with empty corpus returns abstention."""
    async with app_client as client:
        r = await client.post("/query", json={"question": "What is anything?"})
        assert r.status_code == 200
        data = r.json()
        assert data["abstained"] is True
        assert data["abstention_reason"] in ("empty_corpus", "no_relevant_evidence")


@pytest.mark.asyncio
async def test_query_with_data(app_client):
    """POST /query after uploading a document returns a response."""
    async with app_client as client:
        # Upload first
        content = (
            b"The refund policy allows returns within 30 days of purchase. "
            b"Customers must submit refund requests through the customer portal. "
            b"Refunds are processed within 5-7 business days."
        )
        upload_r = await client.post(
            "/documents/upload",
            files={"file": ("policy.txt", content, "text/plain")},
        )
        assert upload_r.status_code == 201

        # Query
        r = await client.post("/query", json={"question": "What is the refund policy?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "request_id" in data
        assert "abstained" in data
        assert "citations" in data


@pytest.mark.asyncio
async def test_query_invalid_request(app_client):
    """POST /query with invalid request returns 422."""
    async with app_client as client:
        r = await client.post("/query", json={})
        assert r.status_code == 422


@pytest.mark.asyncio
async def test_upload_and_delete_document(app_client):
    """Upload then delete a document."""
    async with app_client as client:
        # Upload
        content = b"Some content for deletion test document file."
        upload_r = await client.post(
            "/documents/upload",
            files={"file": ("deleteme.txt", content, "text/plain")},
        )
        assert upload_r.status_code == 201
        doc_id = upload_r.json()["document_id"]

        # Delete
        del_r = await client.delete(f"/documents/{doc_id}")
        assert del_r.status_code == 200
        assert del_r.json()["status"] == "deleted"

        # Confirm gone
        get_r = await client.get(f"/documents/{doc_id}")
        assert get_r.status_code == 404


@pytest.mark.asyncio
async def test_duplicate_upload_replaces(app_client):
    """Uploading the same file replaces the existing document."""
    async with app_client as client:
        content = b"Duplicate test content for testing duplicate detection."
        # First upload
        r1 = await client.post(
            "/documents/upload",
            files={"file": ("dup.txt", content, "text/plain")},
        )
        assert r1.status_code == 201
        doc_id_1 = r1.json()["document_id"]

        # Second upload (same content, same name)
        r2 = await client.post(
            "/documents/upload",
            files={"file": ("dup.txt", content, "text/plain")},
        )
        assert r2.status_code == 201
        doc_id_2 = r2.json()["document_id"]

        # Same document ID due to deterministic generation
        assert doc_id_1 == doc_id_2


@pytest.mark.asyncio
async def test_swagger_docs(app_client):
    """GET /docs returns the Swagger UI."""
    async with app_client as client:
        r = await client.get("/docs")
        assert r.status_code == 200
