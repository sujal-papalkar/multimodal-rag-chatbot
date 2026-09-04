import uuid
import base64
import asyncio
import logging
from typing import List
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document

from app.config import MAX_TABLE_HTML_CHARS
from app.services.llm_service import create_ai_enhanced_summary

logger = logging.getLogger(__name__)


def partition_document(file_path: str):
    logger.info(f"📄 Partitioning document: {file_path}")
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        logger.info(f"✅ Extracted {len(elements)} elements")
        return elements
    except Exception as e:
        logger.warning(f"⚠️ hi_res partitioning failed, retrying with 'fast' strategy: {e}")
        try:
            # EDGE CASE: hi_res partitioning (layout model + OCR) can fail on
            # unusual/scanned/damaged PDFs even when the file itself is
            # valid. Fall back to the lighter 'fast' strategy rather than
            # failing the whole upload outright.
            elements = partition_pdf(
                filename=file_path,
                strategy="fast",
                infer_table_structure=True,
            )
            logger.info(f"✅ Extracted {len(elements)} elements via fallback 'fast' strategy")
            return elements
        except Exception as e2:
            logger.error(f"❌ Failed to partition document {file_path} even with fallback: {e2}")
            raise


def extract_page_images(file_path: str) -> List[str]:
    logger.info("🖼️ Extracting page images...")
    page_images_b64 = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                page_images_b64.append(img_b64)
            except Exception as page_err:
                # EDGE CASE: one malformed page shouldn't kill image
                # extraction for the whole document — append a placeholder
                # marker (None-safe) so page numbering downstream still lines up.
                logger.warning(f"⚠️ Failed to render page {page_num + 1}: {page_err}")
                page_images_b64.append("")
        doc.close()
        logger.info(f"✅ Extracted {len(page_images_b64)} page images")
    except Exception as e:
        logger.warning(f"❌ Failed to extract page images: {e}")
    return page_images_b64


def create_chunks_by_title(elements):
    logger.info("🔨 Creating smart chunks...")
    if not elements:
        # EDGE CASE: a PDF that partitions to zero elements (e.g. a fully
        # blank scanned page, or an image-only page OCR couldn't read)
        # should produce zero chunks cleanly, not crash chunk_by_title.
        logger.warning("⚠️ No elements extracted from document; returning zero chunks.")
        return []
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    logger.info(f"✅ Created {len(chunks)} chunks")
    return chunks


def separate_content_types(chunk):
    content_data = {
        'text': chunk.text,
        'tables': [],
        'table_pages': [],   # page number for each table, parallel to 'tables'
        'images': [],
        'image_pages': [],   # page number for each image, parallel to 'images'
        'types': ['text'],
        'page_number': None,  # kept for backward compat: first element's page
        'page_start': None,
        'page_end': None,
    }

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        orig_elements = chunk.metadata.orig_elements or []

        if orig_elements:
            first_el = orig_elements[0]
            if hasattr(first_el, 'metadata') and hasattr(first_el.metadata, 'page_number'):
                content_data['page_number'] = first_el.metadata.page_number

        # A chunk_by_title chunk can span multiple physical PDF pages (up to
        # max_characters=3000). Recording only the first element's page means
        # any table/image that actually landed on a LATER page within the
        # same chunk gets mis-cited as being on the earlier page. Instead,
        # track the true page range and each element's own page number.
        all_pages = [
            el.metadata.page_number
            for el in orig_elements
            if hasattr(el, 'metadata') and getattr(el.metadata, 'page_number', None) is not None
        ]
        if all_pages:
            content_data['page_start'] = min(all_pages)
            content_data['page_end'] = max(all_pages)

        for element in orig_elements:
            element_type = type(element).__name__
            element_page = getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None

            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                # EDGE CASE: extremely large / malformed tables can blow up
                # the AI-summary prompt and, later, the answer-generation
                # prompt. Cap each table's HTML at ingestion time.
                if table_html and len(table_html) > MAX_TABLE_HTML_CHARS:
                    table_html = table_html[:MAX_TABLE_HTML_CHARS] + "\n...[table truncated]..."
                content_data['tables'].append(table_html)
                content_data['table_pages'].append(element_page)
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
                    content_data['image_pages'].append(element_page)

    content_data['types'] = list(set(content_data['types']))
    return content_data


async def summarise_chunks(chunks):
    """
    Processes all chunks, creating AI summaries in parallel for chunks
    with tables or images.
    """
    logger.info("🧠 Processing chunks with AI Summaries...")
    langchain_documents = []
    chunk_id_to_content_map = {}
    total_chunks = len(chunks)

    if total_chunks == 0:
        # EDGE CASE: zero-chunk documents (blank/unreadable PDFs) should
        # return cleanly rather than the caller assuming non-empty results.
        logger.warning("⚠️ No chunks to summarise.")
        return [], {}

    tasks = []
    chunk_data_list = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{uuid.uuid4()}"
        content_data = separate_content_types(chunk)

        chunk_data_list.append({
            "chunk_id": chunk_id,
            "content_data": content_data
        })

        if content_data['tables'] or content_data['images']:
            logger.info(f"   [Task Created] AI summary for chunk {i + 1}/{total_chunks}")
            tasks.append(create_ai_enhanced_summary(
                content_data['text'],
                content_data['tables'],
                content_data['images']
            ))
        else:
            tasks.append(asyncio.sleep(0, result=content_data['text']))

    logger.info(f"🚀 Running {len(tasks)} summary tasks in parallel...")
    # EDGE CASE: gather with return_exceptions so one failed summary task
    # doesn't take down the whole batch.
    enhanced_content_results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("✅ All summary tasks complete.")

    for i, chunk_data in enumerate(chunk_data_list):
        chunk_id = chunk_data["chunk_id"]
        content_data = chunk_data["content_data"]
        page_number = content_data.get('page_number')

        result = enhanced_content_results[i]
        if isinstance(result, Exception):
            logger.warning(f"⚠️ Summary task {i} raised: {result}; falling back to raw text.")
            enhanced_content = content_data['text'] or "[No extractable text on this page]"
        else:
            enhanced_content = result or content_data['text'] or "[No extractable text on this page]"

        chunk_id_to_content_map[chunk_id] = {
            "raw_text": content_data['text'],
            "tables_html": content_data['tables'],
            "table_pages": content_data.get('table_pages', []),
            "images_base64": content_data['images'],
            "image_pages": content_data.get('image_pages', []),
            "page_number": page_number,
            "page_start": content_data.get('page_start'),
            "page_end": content_data.get('page_end'),
            "ai_summary": enhanced_content,
            "chunk_index": i,
        }

        doc = Document(
            page_content=enhanced_content,
            metadata={"chunk_id": chunk_id, "page_number": page_number}
        )
        langchain_documents.append(doc)

    logger.info(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents, chunk_id_to_content_map
