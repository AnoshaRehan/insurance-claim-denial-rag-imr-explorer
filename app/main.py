import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import router
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="IMR RAG",
    version="0.1.0",
    description="Retrieval-augmented Q&A over California Independent Medical Review determinations.",
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root to the interactive API docs."""
    return RedirectResponse(url="/docs")
