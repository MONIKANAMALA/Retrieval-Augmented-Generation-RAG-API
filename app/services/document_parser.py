"""Document parser: extracts text from PDF, TXT, and Markdown files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from app.core.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class ParsedPage(NamedTuple):
    """A single page / section of extracted text."""
    text: str
    page_number: int | None  # None for non-paginated formats


class ParsedDocument(NamedTuple):
    """Result of parsing a document."""
    pages: list[ParsedPage]
    total_characters: int
    page_count: int | None  # None for non-paginated formats


def validate_extension(filename: str) -> str:
    """Return the lowered extension or raise UnsupportedFileTypeError."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return ext


def parse_pdf(content: bytes) -> ParsedDocument:
    """Extract text from a PDF using PyMuPDF, page by page."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise DocumentParsingError("PyMuPDF is not installed. Install it with: pip install pymupdf") from exc

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise DocumentParsingError(f"Cannot open PDF: {exc}") from exc

    pages: list[ParsedPage] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text") or ""
        text = text.strip()
        if text:
            pages.append(ParsedPage(text=text, page_number=page_num + 1))
    page_count = len(doc)
    doc.close()

    if not pages:
        raise EmptyDocumentError("PDF contains no extractable text.")

    total_chars = sum(len(p.text) for p in pages)
    return ParsedDocument(pages=pages, total_characters=total_chars, page_count=page_count)


def parse_text(content: bytes, filename: str) -> ParsedDocument:
    """Extract text from a plain-text or Markdown file."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception as exc:
            raise DocumentParsingError(f"Cannot decode {filename}: {exc}") from exc

    text = text.strip()
    if not text:
        raise EmptyDocumentError(f"File '{filename}' contains no extractable text.")

    pages = [ParsedPage(text=text, page_number=None)]
    return ParsedDocument(pages=pages, total_characters=len(text), page_count=None)


def parse_document(content: bytes, filename: str) -> ParsedDocument:
    """Parse a document based on its extension.

    Returns a ParsedDocument with extracted pages.
    Raises UnsupportedFileTypeError, EmptyDocumentError, or DocumentParsingError.
    """
    ext = validate_extension(filename)
    logger.info("Parsing document '%s' (type=%s, size=%d bytes)", filename, ext, len(content))

    if ext == ".pdf":
        return parse_pdf(content)
    else:
        return parse_text(content, filename)
