"""Tests for citation validation service."""

from app.services.citations import validate_citations
from app.services.generation import GeneratedCitation


class TestCitationValidation:
    """Tests for citation validation logic."""

    def _make_chunks(self):
        """Create sample retrieved chunks."""
        return [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "filename": "policy.md",
                "chunk_index": 0,
                "page_number": None,
                "text": "All customers are eligible for a full refund within 30 days of purchase.",
                "distance": 0.5,
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "doc_2",
                "filename": "handbook.md",
                "chunk_index": 0,
                "page_number": 3,
                "text": "Employees receive 20 days of annual leave per year. Leave accrues monthly.",
                "distance": 0.6,
            },
        ]

    def test_valid_citation(self):
        """A citation with correct chunk_id and matching text is accepted."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="chunk_1",
                quoted_text="full refund within 30 days",
            )
        ]
        valid, rejected = validate_citations(gen_citations, chunks)
        assert len(valid) == 1
        assert len(rejected) == 0
        assert valid[0].citation_id == "C1"
        assert valid[0].document_id == "doc_1"
        assert valid[0].filename == "policy.md"

    def test_invalid_chunk_id(self):
        """A citation with a non-existent chunk_id is rejected."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="invented_id",
                quoted_text="some text",
            )
        ]
        valid, rejected = validate_citations(gen_citations, chunks)
        assert len(valid) == 0
        assert len(rejected) == 1
        assert "not found" in rejected[0]

    def test_quoted_text_not_in_chunk(self):
        """A citation with text not in the source chunk is rejected."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="chunk_1",
                quoted_text="This text is completely fabricated by the model.",
            )
        ]
        valid, rejected = validate_citations(gen_citations, chunks)
        assert len(valid) == 0
        assert len(rejected) == 1

    def test_empty_quoted_text(self):
        """A citation with empty quoted text is rejected."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="chunk_1",
                quoted_text="",
            )
        ]
        valid, rejected = validate_citations(gen_citations, chunks)
        assert len(valid) == 0
        assert len(rejected) == 1

    def test_whitespace_normalization(self):
        """Citation validation normalizes whitespace in quotes."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="chunk_1",
                quoted_text="full  refund   within  30  days",  # Extra whitespace
            )
        ]
        valid, _rejected = validate_citations(gen_citations, chunks)
        assert len(valid) == 1

    def test_multiple_citations(self):
        """Multiple valid citations are all accepted."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="chunk_1",
                quoted_text="full refund within 30 days",
            ),
            GeneratedCitation(
                chunk_id="chunk_2",
                quoted_text="20 days of annual leave",
            ),
        ]
        valid, _rejected = validate_citations(gen_citations, chunks)
        assert len(valid) == 2
        assert valid[0].citation_id == "C1"
        assert valid[1].citation_id == "C2"

    def test_mixed_valid_and_invalid(self):
        """Mix of valid and invalid citations."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="chunk_1",
                quoted_text="full refund within 30 days",
            ),
            GeneratedCitation(
                chunk_id="fake_id",
                quoted_text="invented text",
            ),
        ]
        valid, rejected = validate_citations(gen_citations, chunks)
        assert len(valid) == 1
        assert len(rejected) == 1

    def test_citation_uses_trusted_metadata(self):
        """Citation metadata comes from stored chunk, not model."""
        chunks = self._make_chunks()
        gen_citations = [
            GeneratedCitation(
                chunk_id="chunk_2",
                quoted_text="20 days of annual leave",
            )
        ]
        valid, _ = validate_citations(gen_citations, chunks)
        assert len(valid) == 1
        # These should come from the chunk metadata, not model
        assert valid[0].document_id == "doc_2"
        assert valid[0].filename == "handbook.md"
        assert valid[0].page_number == 3

    def test_no_citations(self):
        """Empty citation list returns empty results."""
        chunks = self._make_chunks()
        valid, rejected = validate_citations([], chunks)
        assert len(valid) == 0
        assert len(rejected) == 0
