import sqlite3
import time
import base64
import logging
from typing import List, Optional
from app.config import DB_PATH, STORAGE_DIR, STALE_PROCESSING_SECONDS

logger = logging.getLogger(__name__)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content_hash TEXT UNIQUE NOT NULL,
            upload_timestamp REAL NOT NULL,
            num_pages INTEGER,
            status TEXT NOT NULL DEFAULT 'processing',
            error_message TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            page_number INTEGER,
            page_start INTEGER,
            page_end INTEGER,
            raw_text TEXT,
            ai_summary TEXT,
            tables_html TEXT,
            table_pages TEXT,
            image_pages TEXT,
            chunk_index INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    # --- Conversation persistence ---
    # A conversation groups a sequence of messages. Each message is either
    # role='user' (the question) or role='assistant' (the answer). Citations
    # and page_images are stored as JSON *on the assistant message itself*,
    # not in any shared/global table — this is what guarantees each
    # answer's citations stay scoped to that exact answer and never bleed
    # into a different query's results.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            title TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,              -- 'user' | 'assistant'
            content TEXT NOT NULL,
            citations TEXT,                  -- JSON array, assistant messages only
            page_images TEXT,                -- JSON array, assistant messages only
            doc_ids TEXT,                    -- JSON array, which docs this turn used
            created_at REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at)")
    conn.commit()

    # --- Migration for existing DBs created before newer columns existed.
    # CREATE TABLE IF NOT EXISTS won't add columns to an already-existing
    # table, so add them here if missing.
    existing_doc_cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "error_message" not in existing_doc_cols:
        logger.info("🔧 Migrating documents table: adding column 'error_message'")
        conn.execute("ALTER TABLE documents ADD COLUMN error_message TEXT")

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    migrations = {
        "page_start": "ALTER TABLE chunks ADD COLUMN page_start INTEGER",
        "page_end": "ALTER TABLE chunks ADD COLUMN page_end INTEGER",
        "table_pages": "ALTER TABLE chunks ADD COLUMN table_pages TEXT",
        "image_pages": "ALTER TABLE chunks ADD COLUMN image_pages TEXT",
    }
    for col, ddl in migrations.items():
        if col not in existing_cols:
            logger.info(f"🔧 Migrating chunks table: adding column '{col}'")
            conn.execute(ddl)
    conn.commit()
    conn.close()


def reconcile_stale_documents():
    """
    EDGE CASE: if the server crashes/restarts mid-upload, a document can be
    left stuck in status='processing' forever, with no chunks and no way
    for the user to know it failed. On startup, sweep any 'processing' rows
    older than STALE_PROCESSING_SECONDS to 'failed' so they surface as
    actionable (re-uploadable) instead of silently invisible.
    """
    conn = get_db()
    try:
        cutoff = time.time() - STALE_PROCESSING_SECONDS
        stale = conn.execute(
            "SELECT doc_id, filename FROM documents WHERE status = 'processing' AND upload_timestamp < ?",
            (cutoff,)
        ).fetchall()
        if stale:
            conn.execute(
                "UPDATE documents SET status = 'failed', error_message = ? "
                "WHERE status = 'processing' AND upload_timestamp < ?",
                ("Processing was interrupted (server restart). Please re-upload.", cutoff)
            )
            conn.commit()
            for row in stale:
                logger.warning(f"⚠️ Marked stuck document as failed: {row['doc_id']} ({row['filename']})")
    finally:
        conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_page_images(doc_id: str, page_images_b64: List[str]):
    doc_dir = STORAGE_DIR / doc_id
    doc_dir.mkdir(exist_ok=True)
    for i, img_b64 in enumerate(page_images_b64):
        (doc_dir / f"page_{i + 1}.png").write_bytes(base64.b64decode(img_b64))


def load_page_image(doc_id: str, page_number: int) -> Optional[str]:
    path = STORAGE_DIR / doc_id / f"page_{page_number}.png"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")
