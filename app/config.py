"""Application configuration loaded from environment variables."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "imr_determinations"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Retrieval
    top_k: int = 5

    # LLM provider — Groq for everything
    llm_provider: Literal["groq"] = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()