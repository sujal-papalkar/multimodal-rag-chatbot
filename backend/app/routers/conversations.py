import json
import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import ConversationMessage, ConversationHistoryResponse
from app.db.database import get_db
from app.services.conversation_service import ensure_conversation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversations"])


# --- /conversations Endpoint (start a new, empty conversation explicitly) ---
@router.post("/conversations")
async def create_conversation():
    conn = get_db()
    try:
        conversation_id = ensure_conversation(conn, None)
        conn.commit()
        return {"conversation_id": conversation_id}
    finally:
        conn.close()


# --- /conversations/{conversation_id} Endpoint (reload full history + per-message citations) ---
@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(conversation_id: str):
    conn = get_db()
    try:
        conv = conn.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        rows = conn.execute(
            "SELECT message_id, role, content, citations, page_images, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,)
        ).fetchall()

        messages = []
        for row in rows:
            # EDGE CASE: citations/page_images are parsed straight from THIS
            # row's own JSON column — every message's sources are read back
            # exactly as scoped as they were written, per-message.
            messages.append(ConversationMessage(
                message_id=row["message_id"],
                role=row["role"],
                content=row["content"],
                citations=json.loads(row["citations"]) if row["citations"] else [],
                page_images=json.loads(row["page_images"]) if row["page_images"] else [],
                created_at=row["created_at"],
            ))

        return ConversationHistoryResponse(conversation_id=conversation_id, messages=messages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversation")
    finally:
        conn.close()
