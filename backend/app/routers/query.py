from fastapi import APIRouter

from app.models.schemas import QueryRequest, QueryResponse
from app.db.database import get_db
from app.services.conversation_service import (
    ensure_conversation,
    get_recent_history,
    save_message,
)
from app.services.llm_service import condense_followup_query
from app.services.search_service import _run_search_strategy

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_index_api(request: QueryRequest):
    if not request.query or not request.query.strip():
        # EDGE CASE: empty query — not worth creating a conversation/message
        # for, just short-circuit. Echo back whatever conversation_id was
        # sent (or empty) so the frontend's state doesn't get disrupted.
        return QueryResponse(
            answer="Please enter a question.",
            citations=[],
            page_images=[],
            conversation_id=request.conversation_id or "",
            message_id=""
        )

    # --- 1. Resolve conversation + record the user's turn ---
    conn = get_db()
    try:
        conversation_id = ensure_conversation(conn, request.conversation_id)
        history = get_recent_history(conn, conversation_id)
        save_message(conn, conversation_id, "user", request.query, doc_ids=request.doc_ids)
        conn.commit()
    finally:
        conn.close()

    # --- 2. Condense follow-ups using history, but keep retrieval fresh ---
    # EDGE CASE: this is the only place conversation history influences the
    # answer. It rewrites e.g. "where does it occur?" into a standalone
    # question so retrieval (vector search, page lookup, whole-doc intent
    # detection) has something concrete to search for. Retrieval itself
    # always re-runs against the documents for THIS query — nothing about
    # citations is carried over from earlier turns.
    effective_query = await condense_followup_query(history, request.query)
    search_request = QueryRequest(
        query=effective_query,
        doc_ids=request.doc_ids,
        is_web_search_enabled=request.is_web_search_enabled,
        conversation_id=conversation_id,
    )

    # --- 3. Run retrieval + answer generation (produces its own fresh citations) ---
    result = await _run_search_strategy(search_request)

    # --- 4. Persist the assistant's turn with ITS OWN citations only ---
    # EDGE CASE: citations are serialized and stored on this exact message
    # row. Nothing here reads or merges citations from any other message,
    # which is what guarantees Query 2's citations can never include
    # anything from Query 1.
    conn = get_db()
    try:
        assistant_message_id = save_message(
            conn,
            conversation_id,
            "assistant",
            result.answer,
            citations=[c.dict() for c in result.citations],
            page_images=[p.dict() for p in result.page_images],
            doc_ids=request.doc_ids,
        )
        conn.commit()
    finally:
        conn.close()

    result.conversation_id = conversation_id
    result.message_id = assistant_message_id
    return result
