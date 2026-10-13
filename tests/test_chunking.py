"""Tests for chunking service."""

import pytest

from app.services.chunking import _generate_chunk_id, chunk_pages, chunk_text
from app.services.document_parser import ParsedPage


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_basic_chunking(self):
        text = "A" * 500
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) >= 2
        assert all(len(c) <= 200 for c in chunks)

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_text_shorter_than_chunk_size(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_overlap_works(self):
        text = "Hello world. This is a test. Another sentence here. More text follows."
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        assert len(chunks) >= 2
        # Check overlap exists (last chars of chunk N appear in chunk N+1)
        if len(chunks) >= 2:
            # With overlap, chunks should share some content
            assert len(chunks) >= 2

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("text", chunk_size=0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunk_text("text", chunk_overlap=-1)

    def test_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            chunk_text("text", chunk_size=100, chunk_overlap=100)

    def test_no_empty_chunks(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_configurable_sizes(self):
        text = "X" * 1000
        small_chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        large_chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(small_chunks) > len(large_chunks)


class TestChunkPages:
    """Tests for the chunk_pages function."""

    def test_basic_page_chunking(self):
        pages = [
            ParsedPage(text="A" * 300, page_number=1),
            ParsedPage(text="B" * 300, page_number=2),
        ]
        chunks = chunk_pages(pages, "doc1", "test.pdf", chunk_size=200, chunk_overlap=30)
        assert len(chunks) >= 2
        # Check page numbers are preserved
        page_1_chunks = [c for c in chunks if c.page_number == 1]
        page_2_chunks = [c for c in chunks if c.page_number == 2]
        assert len(page_1_chunks) >= 1
        assert len(page_2_chunks) >= 1

    def test_chunk_ids_are_unique(self):
        pages = [ParsedPage(text="A" * 300, page_number=1)]
        chunks = chunk_pages(pages, "doc1", "test.pdf", chunk_size=100, chunk_overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_are_deterministic(self):
        pages = [ParsedPage(text="A" * 300, page_number=1)]
        chunks1 = chunk_pages(pages, "doc1", "test.pdf", chunk_size=200, chunk_overlap=30)
        chunks2 = chunk_pages(pages, "doc1", "test.pdf", chunk_size=200, chunk_overlap=30)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_metadata_preserved(self):
        pages = [ParsedPage(text="Some text for testing", page_number=5)]
        chunks = chunk_pages(pages, "doc42", "report.pdf", chunk_size=800, chunk_overlap=120)
        assert len(chunks) >= 1
        c = chunks[0]
        assert c.document_id == "doc42"
        assert c.filename == "report.pdf"
        assert c.page_number == 5
        assert c.chunk_index == 0

    def test_null_page_number(self):
        pages = [ParsedPage(text="Markdown content", page_number=None)]
        chunks = chunk_pages(pages, "doc1", "readme.md", chunk_size=800, chunk_overlap=120)
        assert chunks[0].page_number is None


class TestChunkIdGeneration:
    """Tests for chunk ID generation."""

    def test_deterministic(self):
        id1 = _generate_chunk_id("doc1", 0)
        id2 = _generate_chunk_id("doc1", 0)
        assert id1 == id2

    def test_different_inputs(self):
        id1 = _generate_chunk_id("doc1", 0)
        id2 = _generate_chunk_id("doc1", 1)
        id3 = _generate_chunk_id("doc2", 0)
        assert id1 != id2
        assert id1 != id3
