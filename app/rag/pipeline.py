from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.rag.embedder import embed_text
from app.rag.generator import generate
from app.rag.prompt import build_prompt
from app.rag.vectorstore import search

logger = logging.getLogger(__name__)


def answer_question(
    question: str,
    top_k: int | None = None,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the full RAG pipeline for a question.

    Args:
        question: The natural-language question to answer
        top_k: Number of records to retrieve. Defaults to settings.top_k
        filters: Optional metadata filters for the vector search

    Returns:
        Dict with the answer text and the source records used.
    """
    settings = get_settings()
    k = top_k or settings.top_k

    logger.info("Embedding question: %s", question)
    question_vector = embed_text(question)

    logger.info("Retrieving top %d records", k)
    retrieved = search(question_vector, top_k=k, filters=filters)

    # Filter out low-confidence matches
    relevant = [r for r in retrieved if r["score"] >= settings.min_retrieval_score]
    filtered_out = len(retrieved) - len(relevant)
    if filtered_out:
        logger.info(
            "Filtered out %d / %d records below min score %.2f",
            filtered_out,
            len(retrieved),
            settings.min_retrieval_score,
        )

    if not relevant:
        logger.warning(
            "No records met the relevance threshold (top score: %.3f). "
            "Question may be outside the dataset.",
            retrieved[0]["score"] if retrieved else 0.0,
        )

    logger.info("Building prompt and calling LLM")
    system_prompt, user_prompt = build_prompt(question, relevant)
    answer = generate(system_prompt, user_prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "reference_id": r["payload"].get("reference_id"),
                "diagnosis_category": r["payload"].get("diagnosis_category"),
                "determination": r["payload"].get("determination"),
                "score": r["score"],
            }
            for r in relevant
        ],
        "low_confidence": len(relevant) == 0,
    }
