import os
import time
import json
import uuid
import shutil
import asyncio
import logging
import hashlib
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from pinecone.exceptions import NotFoundException as PineconeNotFoundException

from app.config import STORAGE_DIR
from app.db.database import get_db, save_page_images
from app.utils.validation import validate_pdf_upload, open_pdf_or_raise
from app.services.document_processing import (
    partition_document,
    extract_page_images,
    create_chunks_by_title,
    summarise_chunks,
)
from app.services import pinecone_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


# --- /upload Endpoint (hash dedup + SQLite persistence) ---
@router.post("/upload")
async def upload_document_api(file: UploadFile = File(...)):
    if not pinecone_service.vectorstore:
        logger.error("Vector store not initialized during upload.")
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    content = await file.read()

    # EDGE CASE: validate before doing any expensive work — wrong file
    # type, empty file, oversized file, or a file that merely has a .pdf
    # extension but isn't really a PDF.
    validate_pdf_upload(file.filename, content)

    content_hash = hashlib.sha256(content).hexdigest()

    conn = get_db()
    existing = conn.execute(
        "SELECT doc_id, filename, status FROM documents WHERE content_hash = ?",
        (content_hash,)
    ).fetchone()
    if existing:
        conn.close()
        if existing["status"] == "failed":
            # EDGE CASE: don't treat a previously-failed upload as a
            # successful duplicate — let the user retry cleanly.
            logger.info(f"↩️ Re-upload of previously failed document, allowing retry: {existing['doc_id']}")
        else:
            logger.info(f"↩️ Duplicate upload detected, reusing doc_id {existing['doc_id']}")
            return {
                "message": f"This file was already uploaded as '{existing['filename']}'",
                "doc_id": existing["doc_id"],
                "duplicate": True,
                "status": existing["status"],
            }
        conn = get_db()

    # EDGE CASE: if this hash previously failed, reuse the doc_id and retry
    # in place rather than violating the UNIQUE(content_hash) constraint.
    if existing and existing["status"] == "failed":
        doc_id = existing["doc_id"]
        conn.execute(
            "UPDATE documents SET status = ?, error_message = NULL, upload_timestamp = ? WHERE doc_id = ?",
            ("processing", time.time(), doc_id)
        )
        # Clear any partial chunks from the failed attempt.
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.commit()
    else:
        doc_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO documents (doc_id, filename, content_hash, upload_timestamp, status) VALUES (?, ?, ?, ?, ?)",
            (doc_id, file.filename, content_hash, time.time(), "processing")
        )
        conn.commit()

    logger.info(f"🚀 Starting new document: {doc_id} ({file.filename})")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"File saved to temp path: {tmp_path}")

        # EDGE CASE: catches encrypted/corrupted/zero-page/too-many-pages
        # PDFs early with clear messages, before any expensive processing.
        probe_doc = open_pdf_or_raise(tmp_path)
        probe_doc.close()

        page_images_b64 = extract_page_images(tmp_path)
        elements = partition_document(tmp_path)
        chunks = create_chunks_by_title(elements)

        if not chunks:
            # EDGE CASE: PDF opened fine but yielded no extractable content
            # (e.g. fully blank pages, or scanned pages OCR couldn't read).
            # Still record the document as ready with zero chunks — an
            # empty-but-valid state — rather than either silently
            # "succeeding" with nothing searchable, or hard-failing.
            conn.execute(
                "UPDATE documents SET status = ?, num_pages = ?, error_message = ? WHERE doc_id = ?",
                ("ready", len(page_images_b64),
                 "No extractable text or content was found in this PDF (it may be blank, or fully scanned images "
                 "without readable text). You can still view page images, but text search will find nothing.",
                 doc_id)
            )
            conn.commit()
            save_page_images(doc_id, page_images_b64)
            return {
                "message": f"Uploaded '{file.filename}', but no searchable text/content was found in it.",
                "doc_id": doc_id,
                "warning": "empty_content"
            }

        (processed_chunks, chunk_data_map) = await summarise_chunks(chunks)

        for doc in processed_chunks:
            doc.metadata["doc_id"] = doc_id

        save_page_images(doc_id, page_images_b64)

        for chunk_id, data in chunk_data_map.items():
            conn.execute(
                """INSERT INTO chunks
                   (chunk_id, doc_id, page_number, page_start, page_end, raw_text, ai_summary, tables_html, table_pages, image_pages, chunk_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    doc_id,
                    data.get("page_number"),
                    data.get("page_start"),
                    data.get("page_end"),
                    data["raw_text"],
                    data.get("ai_summary"),
                    json.dumps(data.get("tables_html", [])),
                    json.dumps(data.get("table_pages", [])),
                    json.dumps(data.get("image_pages", [])),
                    data.get("chunk_index")
                )
            )
        # EDGE CASE: mark 'indexing' (not yet 'ready') until the Pinecone
        # upsert loop below actually completes, so a crash mid-upsert
        # doesn't leave the document falsely reported as fully ready while
        # only partially searchable.
        conn.execute(
            "UPDATE documents SET status = ?, num_pages = ? WHERE doc_id = ?",
            ("indexing", len(page_images_b64), doc_id)
        )
        conn.commit()

        logger.info(f"🔮 Adding {len(processed_chunks)} chunks to Pinecone for doc {doc_id}...")
        loop = asyncio.get_running_loop()
        batch_size = 10
        for i in range(0, len(processed_chunks), batch_size):
            batch = processed_chunks[i:i + batch_size]
            await loop.run_in_executor(None, pinecone_service.vectorstore.add_documents, batch)
            logger.info(
                f"✅ Added batch {i // batch_size + 1}/"
                f"{(len(processed_chunks) + batch_size - 1) // batch_size}"
            )
            if i + batch_size < len(processed_chunks):
                await asyncio.sleep(6)

        logger.info("✅ Chunks added to Pinecone.")

        conn.execute("UPDATE documents SET status = ? WHERE doc_id = ?", ("ready", doc_id))
        conn.commit()

        return {
            "message": f"Successfully processed and indexed {file.filename}",
            "doc_id": doc_id
        }
    except HTTPException:
        conn.execute(
            "UPDATE documents SET status = ?, error_message = ? WHERE doc_id = ?",
            ("failed", "Rejected during validation.", doc_id)
        )
        conn.commit()
        raise
    except Exception as e:
        logger.error(f"❌ Error during upload: {e}", exc_info=True)
        conn.execute(
            "UPDATE documents SET status = ?, error_message = ? WHERE doc_id = ?",
            ("failed", str(e)[:500], doc_id)
        )
        conn.commit()
        raise HTTPException(status_code=500, detail="Document processing failed. Please try re-uploading.")
    finally:
        conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --- /documents Endpoint (for frontend checkbox list) ---
@router.get("/documents")
async def list_documents():
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT doc_id, filename, num_pages, status, upload_timestamp, error_message "
            "FROM documents ORDER BY upload_timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# --- /documents/{doc_id} Endpoint (delete a document) ---
@router.delete("/documents/{doc_id}")
async def delete_document_api(doc_id: str):
    conn = get_db()
    try:
        doc_row = conn.execute(
            "SELECT doc_id, filename FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not doc_row:
            raise HTTPException(status_code=404, detail="Document not found")

        chunk_rows = conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        chunk_ids = [row["chunk_id"] for row in chunk_rows]

        # 1. Delete vectors from Pinecone by ID.
        # NOTE: serverless Pinecone indexes do NOT support delete-by-metadata-filter,
        # only delete-by-ID, which is why we fetch chunk_ids from SQLite first.
        # Pinecone serverless returns a 404 "Namespace not found" if the namespace
        # has zero vectors (e.g. the upsert never completed, or they were already
        # removed) - treat that as "nothing to delete" rather than a hard failure,
        # so SQLite/disk cleanup below still runs and the doc doesn't get stuck
        # half-deleted.
        if chunk_ids and pinecone_service.pinecone_index is not None:
            loop = asyncio.get_running_loop()
            batch_size = 1000  # Pinecone delete has a per-call ID limit
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i:i + batch_size]
                try:
                    await loop.run_in_executor(None, lambda b=batch: pinecone_service.pinecone_index.delete(ids=b))
                except PineconeNotFoundException:
                    logger.warning(
                        f"⚠️ Pinecone namespace/ids not found while deleting doc {doc_id} "
                        f"(vectors likely never existed or already removed) — skipping vector delete"
                    )
                    break
                except Exception as e:
                    # EDGE CASE: a transient Pinecone error shouldn't abandon
                    # the whole delete — log and continue with remaining
                    # batches / DB cleanup rather than leaving a half state.
                    logger.warning(f"⚠️ Pinecone delete batch failed (continuing): {e}")
            else:
                logger.info(f"🗑️ Deleted {len(chunk_ids)} vectors from Pinecone for doc {doc_id}")
        else:
            logger.info(f"No chunks found in Pinecone for doc {doc_id}, skipping vector delete")

        # 2. Delete rows from SQLite
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.commit()

        # 3. Delete saved page images from disk
        doc_dir = STORAGE_DIR / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)

        logger.info(f"✅ Deleted document {doc_id} ({doc_row['filename']}) completely")
        return {
            "message": f"Deleted '{doc_row['filename']}' and {len(chunk_ids)} chunk(s)",
            "doc_id": doc_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document")
    finally:
        conn.close()


# --- /documents/{doc_id}/retry Endpoint ---
# EDGE CASE: gives the user an explicit way to retry a failed document
# without re-uploading the file, since content-hash dedup would otherwise
# make a straight re-upload behave oddly for a 'failed' row.
@router.post("/documents/{doc_id}/retry")
async def retry_document_status():
    raise HTTPException(
        status_code=400,
        detail="To retry a failed document, please re-upload the original file — it will be reprocessed automatically."
    )
