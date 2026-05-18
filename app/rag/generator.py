from __future__ import annotations

import logging
from functools import lru_cache

from groq import Groq

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Return a cached Groq client."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return Groq(api_key=settings.groq_api_key)


def generate(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Generate an answer from the LLM.

    Args:
        system_prompt: System-level instructions
        user_prompt: The user message (records + question)
        temperature: Controls randomness

    Returns:
        The generated answer text.
    """
    settings = get_settings()
    client = get_client()

    logger.info("Sending request to Groq (model=%s)", settings.groq_model)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content or ""
    logger.info(
        "Received %d-character response (input tokens: %d, output tokens: %d)",
        len(answer),
        response.usage.prompt_tokens if response.usage else 0,
        response.usage.completion_tokens if response.usage else 0,
    )
    return answer.strip()
