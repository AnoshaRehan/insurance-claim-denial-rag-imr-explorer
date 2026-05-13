"""FastAPI application entrypoint."""
from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()
app = FastAPI(title="IMR RAG", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "environment": settings.environment}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "IMR RAG API. See /docs for the API documentation."}