import logging
from fastapi import APIRouter, HTTPException

from app.services.ragas_eval import (
    EvalRequest,
    EvalResponse,
    run_ragas_evaluation,
    RAGAS_AVAILABLE,
)
from app.services.search_service import perform_document_search
from app.services.llm_service import llm
from app.services.pinecone_service import embeddings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])


# --- /evaluate Endpoint (RAGAS-based RAG evaluation) ---
# EDGE CASE: evaluation runs through the EXACT SAME perform_document_search
# / generate_final_answer pipeline that serves real traffic (see
# ragas_eval.py), so scores reflect production behavior rather than a
# parallel eval-only implementation that could silently drift out of sync.
@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_api(request: EvalRequest):
    if not RAGAS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ragas is not installed on the server. Run `pip install ragas` and restart to enable evaluation."
        )
    if not request.cases:
        raise HTTPException(status_code=400, detail="Provide at least one case to evaluate.")
    try:
        return await run_ragas_evaluation(request.cases, perform_document_search, llm, embeddings)
    except Exception as e:
        logger.error(f"❌ RAGAS evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
