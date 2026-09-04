import time
import json
import uuid
import sqlite3
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# --- Conversation helpers ---
# EDGE CASE: these are the only functions that read/write conversation
# state. Citations are written ONLY as a JSON blob on the assistant
# message row that produced them (see save_message) — there is no global
# citations table anywhere, so it's structurally impossible for one
# query's citations to leak into another's response.
MAX_HISTORY_MESSAGES = 12  # ~6 turns of user+assistant, bounds prompt size for condensing


def ensure_conversation(conn: sqlite3.Connection, conversation_id: Optional[str]) -> str:
    """Returns a valid conversation_id, creating a new row if none was given or the given id is unknown."""
    now = time.time()
    if conversation_id:
        row = conn.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        if row:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?", (now, conversation_id))
            return conversation_id
        # EDGE CASE: client sent a conversation_id we don't recognize (e.g.
        # cleared server DB, or a stale id from another environment) —
        # create it fresh under that same id rather than erroring, so the
        # frontend doesn't need special-case handling for "unknown id".
        conn.execute(
            "INSERT INTO conversations (conversation_id, created_at, updated_at) VALUES (?, ?, ?)",
            (conversation_id, now, now)
        )
        return conversation_id

    new_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO conversations (conversation_id, created_at, updated_at) VALUES (?, ?, ?)",
        (new_id, now, now)
    )
    return new_id


def save_message(
    conn: sqlite3.Connection,
    conversation_id: str,
    role: str,
    content: str,
    citations: Optional[List[dict]] = None,
    page_images: Optional[List[dict]] = None,
    doc_ids: Optional[List[str]] = None,
) -> str:
    """
    Persists one turn (user question OR assistant answer). Citations and
    page_images, when present, are stored inline on THIS row only — this is
    the mechanism that keeps each answer's sources scoped to that answer.
    """
    message_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO messages
           (message_id, conversation_id, role, content, citations, page_images, doc_ids, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            message_id,
            conversation_id,
            role,
            content,
            json.dumps(citations) if citations is not None else None,
            json.dumps(page_images) if page_images is not None else None,
            json.dumps(doc_ids) if doc_ids is not None else None,
            time.time(),
        )
    )
    return message_id


def get_recent_history(conn: sqlite3.Connection, conversation_id: str) -> List[Dict[str, str]]:
    """Returns recent (role, content) pairs for building conversational context — text only, no citations."""
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,)
    ).fetchall()
    trimmed = rows[-MAX_HISTORY_MESSAGES:] if rows else []
    return [{"role": r["role"], "content": r["content"]} for r in trimmed]
