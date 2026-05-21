from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from app.config import get_settings
from app.rag.embedder import embed_text
from app.rag.pipeline import answer_question
from app.rag.vectorstore import count_points, search

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check that also verifies Qdrant connectivity."""
    settings = get_settings()
    qdrant_connected = False
    indexed_points = None
    try:
        indexed_points = count_points()
        qdrant_connected = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant health check failed: %s", exc)

    return HealthResponse(
        status="ok",
        environment=settings.environment,
        qdrant_connected=qdrant_connected,
        indexed_points=indexed_points,
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run the full RAG pipeline and return an answer with sources."""
    filters = {}
    if request.filter_diagnosis:
        filters["diagnosis_category"] = request.filter_diagnosis

    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            filters=filters or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in /ask")
        raise HTTPException(status_code=500, detail="Failed to generate answer.") from exc

    return AskResponse(**result)


@router.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest) -> SearchResponse:
    """Raw semantic search without LLM generation. Useful for debugging."""
    filters = {}
    if request.filter_diagnosis:
        filters["diagnosis_category"] = request.filter_diagnosis

    try:
        query_vector = embed_text(request.query)
        raw_results = search(query_vector, top_k=request.top_k, filters=filters or None)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in /search")
        raise HTTPException(status_code=500, detail="Search failed.") from exc

    results = [
        {
            "reference_id": r["payload"].get("reference_id"),
            "diagnosis_category": r["payload"].get("diagnosis_category"),
            "diagnosis_subcategory": r["payload"].get("diagnosis_subcategory"),
            "determination": r["payload"].get("determination"),
            "findings": (r["payload"].get("findings") or "")[:500],
            "score": r["score"],
        }
        for r in raw_results
    ]
    return SearchResponse(query=request.query, results=results)
