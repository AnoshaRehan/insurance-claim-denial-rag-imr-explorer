"""Wrapper around sentence-transformers for generating embeddings."""
from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once and reuse it.

    The model is downloaded on first use and cached locally
    by sentence-transformers.
    """
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded")
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of texts into vectors.

    Args:
        texts: List of strings to embed.
        batch_size: How many texts to process at once. Larger is faster
            but uses more memory.

    Returns:
        List of vectors. Each vector is a list of floats.
    """
    model = get_embedder() # cached at ~/.cache/huggingface/
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-similarity search
    )
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    """Convenience wrapper to embed a single text."""
    return embed_texts([text])[0]
