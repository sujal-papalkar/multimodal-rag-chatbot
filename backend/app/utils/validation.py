import logging
import fitz  # PyMuPDF
from fastapi import HTTPException
from app.config import MAX_UPLOAD_BYTES, MAX_PDF_PAGES

logger = logging.getLogger(__name__)


# --- File validation helpers (EDGE CASE: non-PDF, renamed, empty, oversized files) ---
def validate_pdf_upload(filename: str, content: bytes):
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File exceeds the {max_mb:.0f} MB upload limit.")

    # Magic-byte sniff: a file renamed to .pdf but not actually a PDF will
    # fail loudly and clearly here instead of crashing deep inside
    # partition_pdf/fitz with a confusing stack trace.
    if not content[:5].startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF.")


def open_pdf_or_raise(tmp_path: str) -> "fitz.Document":
    """
    Opens a PDF with PyMuPDF, translating common failure modes (encrypted,
    corrupted, zero-page) into clear HTTPExceptions instead of letting a
    raw fitz exception propagate as a generic 500.
    """
    try:
        doc = fitz.open(tmp_path)
    except Exception as e:
        logger.error(f"❌ Failed to open PDF: {e}")
        raise HTTPException(status_code=400, detail="The PDF could not be opened. It may be corrupted.")

    if doc.is_encrypted:
        # Try an empty password first (some "encrypted" PDFs only restrict
        # printing/editing, not viewing, and open fine with "").
        if not doc.authenticate(""):
            doc.close()
            raise HTTPException(
                status_code=400,
                detail="This PDF is password-protected. Please upload an unlocked PDF."
            )

    if doc.page_count == 0:
        doc.close()
        raise HTTPException(status_code=400, detail="This PDF has no pages.")

    if doc.page_count > MAX_PDF_PAGES:
        doc.close()
        raise HTTPException(
            status_code=400,
            detail=f"This PDF has {doc.page_count} pages, which exceeds the {MAX_PDF_PAGES}-page limit."
        )

    return doc
