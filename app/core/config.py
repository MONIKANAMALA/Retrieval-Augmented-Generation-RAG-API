"""Core configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    APP_ENV: str = Field(default="development")
    LLM_PROVIDER: str = Field(default="huggingface")

    # Local Hugging Face Model
    HF_MODEL_NAME: str = Field(default="google/flan-t5-small")

    # Mistral
    MISTRAL_API_KEY: str = Field(default="")
    MISTRAL_MODEL: str = Field(default="mistral-small-latest")

    # Embeddings
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = Field(default="./storage/chroma")
    CHROMA_COLLECTION_NAME: str = Field(default="assessment_documents")

    # Chunking
    CHUNK_SIZE: int = Field(default=800, ge=100)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)

    # Retrieval
    DEFAULT_TOP_K: int = Field(default=5, ge=1)
    MIN_RELEVANCE_SCORE: float = Field(default=1.45, ge=0.0)

    # Upload
    MAX_UPLOAD_SIZE_MB: int = Field(default=10, ge=1)

    # LLM
    LLM_TIMEOUT_SECONDS: int = Field(default=30, ge=5)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def chroma_path(self) -> Path:
        return Path(self.CHROMA_PERSIST_DIRECTORY)

    @property
    def has_mistral_key(self) -> bool:
        return bool(self.MISTRAL_API_KEY and self.MISTRAL_API_KEY.strip())


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
