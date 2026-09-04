import os
import sqlite3
import logging
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage

from app.config import (
    GEMINI_MODEL,
    MAX_ANSWER_PROMPT_CHARS,
    MAX_SUMMARY_DOC_CHARS,
)
from app.db.database import get_db
from app.utils.text_utils import _truncate_contexts_to_budget, _validate_citations

logger = logging.getLogger(__name__)

# --- Global LLM and Serper inits ---
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=os.environ.get("GEMINI_API_KEY")
)
web_search_tool = GoogleSerperAPIWrapper()


def extract_text(response) -> str:
    """
    Safely extract plain text from an LLM response, whether `.content`
    is a plain string or a list of content blocks (text/thinking/
    function_call/thought_signature/etc.). Some Gemini models (notably
    ones with automatic function calling enabled) can return
    `response.content` as a list of dicts instead of a string, which
    breaks any code (e.g. Pydantic models) expecting a plain str.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Only pull actual text blocks; skip function_call,
                # thought_signature, or other non-text metadata blocks.
                if item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
        text = "".join(parts).strip()
        if text:
            return text
        logger.warning(f"⚠️ LLM response had no extractable text content: {content}")
        return ""

    if content is None:
        return ""

    return str(content)


async def condense_followup_query(history: List[Dict[str, str]], query: str) -> str:
    """
    EDGE CASE: a bare follow-up like "where does it occur?" has no lexical
    or semantic anchor of its own — vector retrieval, the page-number
    regex, and the whole-document intent detector all operate on the
    literal query text, and none of them know "it" means "photosynthesis"
    from three turns ago.

    This rewrites such follow-ups into a standalone question using the
    recent conversation, BEFORE retrieval runs. Retrieval below still
    always re-runs fresh against the documents for the rewritten query —
    conversation history is used only to disambiguate *what* to search
    for, never as a substitute for actually searching. This is what keeps
    "maintain conversation history" and "citations must be fresh per
    query" from conflicting with each other.

    Falls back to the original query untouched if there's no history yet,
    or if the rewrite call fails or returns something degenerate — a
    failed rewrite must never block the query.
    """
    if not history:
        return query

    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history[-8:])
    prompt = f"""Given this recent conversation:
{history_text}

And this new question: "{query}"

If the new question depends on the conversation (e.g. uses "it", "that", "there",
or otherwise only makes sense with the prior context), rewrite it as a fully
standalone question that preserves its original intent. If it is already
standalone, return it completely unchanged.

Reply with ONLY the question text — no preamble, no quotes, no explanation."""

    try:
        response = await llm.ainvoke(prompt)
        rewritten = extract_text(response).strip().strip('"').strip()
        # Sanity guard: reject empty or suspiciously long/garbage rewrites.
        if rewritten and 0 < len(rewritten) < 500:
            return rewritten
    except Exception as e:
        logger.warning(f"⚠️ Follow-up query condensing failed, using original query verbatim: {e}")

    return query


async def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Asynchronously generates a summary using the LLM."""
    global llm
    try:
        prompt_text = f"""You are creating a searchable description for a document chunk. 
Your goal is to make this content easily findable through semantic search.

TEXT CONTENT:
{text}
"""
        if tables:
            prompt_text += f"\nTABLES:\n{tables}\n"

        prompt_text += """
YOUR TASK: Generate a comprehensive, searchable description that:
1. Summarizes the main topics and key information
2. Describes what's in any tables (column headers, data types, key findings)
3. Describes what's shown in any images
4. Uses clear, specific terminology that someone would search for
5. Maintains important details like numbers, dates, names

Keep it factual and detailed. This will be used for semantic search retrieval.
"""

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        message = HumanMessage(content=message_content)

        response = await llm.ainvoke([message])
        summary = extract_text(response)
        if not summary:
            # EDGE CASE: LLM returned nothing extractable (e.g. safety
            # filter, empty completion). Fall back to a deterministic
            # summary so the chunk is still searchable.
            summary = f"{text[:300]}... [Contains {len(tables)} table(s)] [Contains {len(images)} image(s)]"
        return summary
    except Exception as e:
        logger.warning(f"     ❌ AI summary failed: {e}")
        return f"{text[:300]}... [Contains {len(tables)} table(s)] [Contains {len(images)} image(s)]"


async def generate_final_answer(chunks: List[Dict], query: str):
    logger.info("🧠 Generating final answer from DOCUMENTS...")
    global llm

    if not chunks:
        # EDGE CASE: defensive guard — callers should already avoid this,
        # but never build an empty prompt.
        return "Sorry, I couldn't find relevant context in the selected documents."

    # EDGE CASE: bound total prompt size by a token/char budget, not just
    # chunk count, so a handful of huge (table-heavy) chunks can't blow the
    # context window or balloon latency/cost.
    chunks = _truncate_contexts_to_budget(chunks, MAX_ANSWER_PROMPT_CHARS)

    prompt_text = f"""Based on the following document excerpts, please answer this question: {query}

CONTENT TO ANALYZE:
"""

    for i, chunk_data in enumerate(chunks):
        source_label = chunk_data.get("filename") or chunk_data.get("doc_id", "Unknown document")
        page_start = chunk_data.get("page_start") or chunk_data.get("page_number")
        page_end = chunk_data.get("page_end") or page_start
        if page_start and page_end and page_start != page_end:
            page_label = f"Pages {page_start}-{page_end}"
        else:
            page_label = f"Page {page_start or 'N/A'}"
        prompt_text += (
            f"--- Document Excerpt {i + 1} "
            f"(from '{source_label}', {page_label}) ---\n"
        )
        if chunk_data.get("raw_text"):
            prompt_text += f"TEXT:\n{chunk_data['raw_text']}\n\n"
        if chunk_data.get("tables_html"):
            prompt_text += "TABLES (source HTML, for your reference only):\n"
            for j, table in enumerate(chunk_data["tables_html"]):
                prompt_text += f"Table {j + 1}:\n{table}\n\n"

    prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images.

FORMATTING RULES FOR TABLES (read carefully):
- Never output raw HTML (no <table>, <tr>, <td>, etc.) in your answer, even if the
  source material above is given to you as HTML.
- When your answer includes tabular data, render it as a GitHub-flavored Markdown
  pipe table, for example:
  | Place | Height (km) | g (m/s²) |
  |---|---|---|
  | Surface of the earth | 0 | 9.8 |
- Keep column headers short and consistent with the source. If a source cell is
  blank or illegible, write "—" rather than leaving it empty or guessing a value.
- Only build a table when the underlying content is actually tabular; otherwise
  write normal prose.

If the provided excerpts do not actually contain information relevant to the
question, say so plainly instead of guessing or inventing an answer.
Cite the source document and page for facts you use, e.g., [DocumentName, Page 5] or
[DocumentName, Pages 5-6] when an excerpt is labeled as spanning multiple pages.
Never state a single page number more precisely than what's given if an excerpt spans
multiple pages — cite the full range instead of guessing which page a detail is on.
Only cite pages that are actually present in the excerpts above — never invent a
page number.

ANSWER:"""

    message_content = [{"type": "text", "text": prompt_text}]

    for chunk_data in chunks:
        for image_base64 in chunk_data.get("images_base64", []):
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

    message = HumanMessage(content=message_content)
    try:
        response = await llm.ainvoke([message])
        answer = extract_text(response)
        if not answer:
            # EDGE CASE: empty completion (safety filter, quota edge, etc.)
            return "Sorry, I received an empty response while generating the answer. Please try rephrasing your question."
        _validate_citations(answer, chunks)
        return answer
    except Exception as e:
        logger.error(f"❌ LLM call failed in generate_final_answer: {e}")
        return "Sorry, I encountered an error while generating the final answer. Please try again in a moment."


async def generate_whole_document_summary(doc_ids: List[str], filenames: Dict[str, str], query: str) -> str:
    """
    EDGE CASE: questions like "summarize this document", "list everything
    in this PDF", or "how many pages/sections/examples are there" are
    poorly served by top-k similarity search, which only returns a handful
    of semantically-similar chunks rather than full document coverage.
    This does a simple map-reduce over ALL chunks for the requested
    document(s): summarize per-document (bounded by a char budget, in
    chunk_index order so the structure is preserved), then combine.
    """
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(doc_ids))
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE doc_id IN ({placeholders}) ORDER BY doc_id, chunk_index",
            doc_ids
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "Sorry, I couldn't find any content for the selected document(s)."

    # Group by doc_id, preserving order.
    by_doc: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        by_doc.setdefault(row["doc_id"], []).append(row)

    per_doc_summaries = []
    for doc_id, doc_rows in by_doc.items():
        filename = filenames.get(doc_id, doc_id)
        combined_text = ""
        for row in doc_rows:
            piece = f"\n[Page {row['page_start'] or row['page_number'] or '?'}] {row['raw_text'] or ''}"
            if len(combined_text) + len(piece) > MAX_SUMMARY_DOC_CHARS:
                combined_text += "\n...[document truncated for length]..."
                break
            combined_text += piece

        prompt = f"""You are summarizing an entire document titled '{filename}' to answer this request: "{query}"

DOCUMENT CONTENT (in page order):
{combined_text}

Provide a thorough, well-organized answer that covers the whole document as given above,
not just one section. Use headings or bullet points if that helps organize the material.
If the request asks for a count (e.g. "how many examples/figures/sections"), count
carefully based only on what's shown above and note if the document was truncated.
If your answer includes tabular data, render it as a GitHub-flavored Markdown pipe
table rather than raw HTML.
"""
        try:
            response = await llm.ainvoke(prompt)
            summary = extract_text(response) or f"[Could not summarize '{filename}']"
        except Exception as e:
            logger.error(f"❌ Whole-document summary failed for {doc_id}: {e}")
            summary = f"[Error summarizing '{filename}']"
        per_doc_summaries.append(f"### {filename}\n{summary}")

    if len(per_doc_summaries) == 1:
        return per_doc_summaries[0]

    # Multiple documents: combine section-by-section rather than a second
    # LLM pass, to avoid compounding truncation/hallucination risk.
    return "\n\n".join(per_doc_summaries)
