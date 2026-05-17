"""Wrapper around Qdrant for storing and searching vectors."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_client() -> QdrantClient:
    """Return a Qdrant client configured from settings."""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(recreate: bool = False) -> None:
    """Create the Qdrant collection if it doesn't exist.

    Args:
        recreate: If True, delete and recreate the collection.
            Useful for re-indexing from scratch.
    """
    settings = get_settings()
    client = get_client()
    collection = settings.qdrant_collection

    exists = client.collection_exists(collection)

    if exists and recreate:
        logger.info("Deleting existing collection: %s", collection)
        client.delete_collection(collection)
        exists = False

    if not exists:
        logger.info("Creating collection: %s", collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )


def upsert_points(
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    batch_size: int = 256,
) -> None:
    """Insert or update points in the collection.

    Args:
        vectors: List of embedding vectors.
        payloads: List of metadata dicts, one per vector.
            Must be the same length as vectors.
        batch_size: How many points to send per Qdrant request.
    """
    if len(vectors) != len(payloads):
        raise ValueError("vectors and payloads must have the same length")

    settings = get_settings()
    client = get_client()
    collection = settings.qdrant_collection

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
        for vec, payload in zip(vectors, payloads, strict=True)
    ]

    # Send in batches to avoid huge single requests
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)
        logger.info("Upserted %d points (%d / %d)", len(batch), i + len(batch), len(points))


def search(
    query_vector: list[float],
    top_k: int = 5,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Find the top_k most similar vectors, optionally filtered by metadata.

    Args:
        query_vector: The vector to search for.
        top_k: How many results to return.
        filters: Optional metadata filters, e.g.,
            {"diagnosis_category": "Mental Disorder"}.

    Returns:
        List of results, each containing 'score' and 'payload'.
    """
    settings = get_settings()
    client = get_client()

    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points

    return [{"score": r.score, "payload": r.payload} for r in results]


def count_points() -> int:
    """Return how many points are in the collection."""
    settings = get_settings()
    client = get_client()
    return client.count(collection_name=settings.qdrant_collection).count