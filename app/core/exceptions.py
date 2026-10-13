"""Custom exception types for the RAG application."""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for RAG application errors."""


class DocumentParsingError(RAGError):
    """Raised when a document cannot be parsed or text extraction fails."""


class UnsupportedFileTypeError(RAGError):
    """Raised when an uploaded file has an unsupported extension."""


class FileTooLargeError(RAGError):
    """Raised when an uploaded file exceeds the maximum size."""


class EmptyDocumentError(RAGError):
    """Raised when a document contains no extractable text."""


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class VectorStoreError(RAGError):
    """Raised when a vector store operation fails."""


class RetrievalError(RAGError):
    """Raised when retrieval fails."""


class GenerationError(RAGError):
    """Raised when LLM answer generation fails."""


class GenerationConfigError(GenerationError):
    """Raised when the LLM provider is not configured (e.g., missing API key)."""


class CitationValidationError(RAGError):
    """Raised when citation validation fails."""
