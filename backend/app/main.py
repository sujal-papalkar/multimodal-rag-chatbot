import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.db.database import init_db, reconcile_stale_documents
from app.services.pinecone_service import init_pinecone
from app.services.ragas_eval import RAGAS_AVAILABLE
from app.routers import (
    documents,
    query,
    conversations,
    evaluation,
    health,
)

logger = logging.getLogger(__name__)


# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    init_pinecone()
    init_db()
    reconcile_stale_documents()
    if not RAGAS_AVAILABLE:
        logger.warning("⚠️ ragas not installed — /evaluate will return 503 until `pip install ragas` is run.")
    yield
    logger.info("Server shutting down...")


# --- FastAPI Instance ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Include Routers ---
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(conversations.router)
app.include_router(evaluation.router)
app.include_router(health.router)


# --- Main Entrypoint ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server at http://localhost:8000")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
