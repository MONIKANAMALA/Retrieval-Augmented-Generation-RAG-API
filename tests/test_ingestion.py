"""Tests for document ingestion: parsing, validation, and storage."""

import pytest

from app.core.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.services.document_parser import (
    parse_document,
    validate_extension,
)


class TestValidateExtension:
    """Tests for file extension validation."""

    def test_valid_extensions(self):
        for ext in [".pdf", ".txt", ".md", ".markdown"]:
            result = validate_extension(f"test{ext}")
            assert result == ext

    def test_case_insensitive(self):
        assert validate_extension("test.TXT") == ".txt"
        assert validate_extension("test.Md") == ".md"

    def test_unsupported_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_extension("test.docx")

    def test_no_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_extension("testfile")

    def test_exe_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_extension("malware.exe")


class TestParseTxt:
    """Tests for TXT/Markdown parsing."""

    def test_valid_txt(self):
        content = b"Hello, this is a test document."
        result = parse_document(content, "test.txt")
        assert result.total_characters > 0
        assert result.page_count is None
        assert len(result.pages) == 1
        assert result.pages[0].page_number is None
        assert "Hello" in result.pages[0].text

    def test_valid_markdown(self):
        content = b"# Title\n\nSome **bold** text."
        result = parse_document(content, "test.md")
        assert result.total_characters > 0
        assert len(result.pages) == 1

    def test_empty_txt(self):
        with pytest.raises(EmptyDocumentError):
            parse_document(b"", "test.txt")

    def test_whitespace_only_txt(self):
        with pytest.raises(EmptyDocumentError):
            parse_document(b"   \n\t  ", "test.txt")

    def test_utf8_content(self):
        content = "Héllo wörld — special chars".encode()
        result = parse_document(content, "test.txt")
        assert "Héllo" in result.pages[0].text


class TestParsePdf:
    """Tests for PDF parsing."""

    def test_valid_pdf(self):
        """Test parsing a minimal valid PDF."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        # Create a minimal test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF content on page 1")
        page2 = doc.new_page()
        page2.insert_text((50, 50), "More content on page 2")
        pdf_bytes = doc.tobytes()
        doc.close()

        result = parse_document(pdf_bytes, "test.pdf")
        assert result.total_characters > 0
        assert result.page_count == 2
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2

    def test_empty_pdf(self):
        """Test parsing a PDF with no text."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        doc = fitz.open()
        doc.new_page()  # Empty page
        pdf_bytes = doc.tobytes()
        doc.close()

        with pytest.raises(EmptyDocumentError):
            parse_document(pdf_bytes, "empty.pdf")

    def test_malformed_pdf(self):
        with pytest.raises(DocumentParsingError):
            parse_document(b"not a real pdf", "bad.pdf")


class TestDuplicateHandling:
    """Tests for duplicate upload behavior."""

    def test_deterministic_document_id(self):
        """Same file content + filename should produce same document ID."""
        from app.api.routes_documents import _generate_document_id
        content = b"Hello test content"
        id1 = _generate_document_id("test.txt", content)
        id2 = _generate_document_id("test.txt", content)
        assert id1 == id2

    def test_different_content_different_id(self):
        from app.api.routes_documents import _generate_document_id
        id1 = _generate_document_id("test.txt", b"content1")
        id2 = _generate_document_id("test.txt", b"content2")
        assert id1 != id2

    def test_different_name_different_id(self):
        from app.api.routes_documents import _generate_document_id
        content = b"same content"
        id1 = _generate_document_id("file1.txt", content)
        id2 = _generate_document_id("file2.txt", content)
        assert id1 != id2
