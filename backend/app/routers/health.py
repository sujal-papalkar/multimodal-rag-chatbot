import logging
from fastapi import APIRouter

from app.db.database import get_db
from app.services import pinecone_service
from app.services.ragas_eval import RAGAS_AVAILABLE

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# --- /health Endpoint (basic liveness/readiness check) ---
@router.get("/health")
async def health_check():
    checks = {"db": False, "pinecone": False, "ragas": RAGAS_AVAILABLE}
    try:
        conn = get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        checks["db"] = True
    except Exception as e:
        logger.warning(f"Health check: DB not reachable: {e}")

    checks["pinecone"] = (
        pinecone_service.vectorstore is not None
        and pinecone_service.pinecone_index is not None
    )

    ok = checks["db"] and checks["pinecone"]  # ragas is optional, doesn't gate readiness
    return {"status": "ok" if ok else "degraded", "checks": checks}
