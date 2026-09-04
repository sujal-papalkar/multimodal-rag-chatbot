import re
import uuid
import json
import asyncio
import sqlite3
import logging
from typing import List, Dict, Any

from app.models.schemas import QueryRequest, QueryResponse, Citation, PageImage
from app.db.database import get_db, load_page_image
from app.services.llm_service import (
    llm,
    web_search_tool,
    extract_text,
    generate_final_answer,
    generate_whole_document_summary,
)
from app.services import pinecone_service
from app.services.pinecone_service import rerank_chunks

logger = logging.getLogger(__name__)

# --- Whole-document intent regex ---
_WHOLE_DOC_INTENT_RE = re.compile(
    r"\b(summar(?:y|ize|ise)|overview|entire document|whole document|"
    r"whole pdf|entire pdf|all pages|list all|list every|how many (?:pages|sections|"
    r"chapters|figures|tables|examples)|table of contents|outline)\b",
    re.IGNORECASE
)


def is_whole_document_query(query: str) -> bool:
    """Detects intents that need full-document coverage rather than top-k retrieval."""
    return bool(_WHOLE_DOC_INTENT_RE.search(query))


# Matches "page 1", "pages 1", "page no. 1", "page number 1" (single page)
_PAGE_SINGLE_RE = re.compile(r'\bpages?\s*(?:no\.?|number)?\s*(\d+)\b', re.IGNORECASE)
# Matches "pg 1", "pg.1"
_PAGE_PG_RE = re.compile(r'\bpg\.?\s*(\d+)\b', re.IGNORECASE)
# Matches "page 1-3", "pages 1 to 3" (ranges)
_PAGE_RANGE_RE = re.compile(r'\bpages?\s*(\d+)\s*(?:-|to)\s*(\d+)\b', re.IGNORECASE)
# EDGE CASE: a bare "p1"/"p 1"/"p.1" pattern is too easily confused with
# ordinary variable names in technical/scientific documents (e.g. physics
# "p = 1", "p1" as momentum-at-state-1). We only treat a bare "p<number>"
# as a page reference when it's immediately preceded by an explicit page
# cue word ("on", "see", "refer to", "from", "at", "check") within a few
# characters, which real page references almost always have and stray
# variable mentions almost never do.
_PAGE_P_CONTEXTUAL_RE = re.compile(
    r'\b(?:on|see|refer(?:\s+to)?|from|at|check|open)\s+p\.?\s*(\d+)\b',
    re.IGNORECASE
)

MAX_PAGE_RANGE_SPAN = 20  # safety cap so "page 1 to 9999" can't blow up the query
MAX_PAGE_MATCH_CHUNKS = 30  # safety cap on how many chunks a page-lookup can return (also budget-capped below)


def extract_requested_pages(query: str) -> List[int]:
    """
    Detects explicit page-number references in a user query (e.g. "explain
    page 1", "what is on page 6", "pages 3-5") so they can be looked up
    directly by page number instead of relying on semantic/vector search —
    which has no concept of page numbers and can easily retrieve unrelated
    content that happens to mention "page" in its text.

    Deliberately conservative about the bare "p<N>" pattern (see
    _PAGE_P_CONTEXTUAL_RE above) since documents with formulas/variables
    (physics, math, engineering PDFs) can otherwise trigger false positives
    like "p = 1" being read as "page 1".
    """
    pages = set()

    for start_s, end_s in _PAGE_RANGE_RE.findall(query):
        start, end = int(start_s), int(end_s)
        if start > end:
            start, end = end, start
        if end - start <= MAX_PAGE_RANGE_SPAN:
            pages.update(range(start, end + 1))

    for pattern in (_PAGE_SINGLE_RE, _PAGE_PG_RE, _PAGE_P_CONTEXTUAL_RE):
        for match in pattern.findall(query):
            pages.add(int(match))

    return sorted(pages)


def _hydrate_chunk_row(row: sqlite3.Row, filenames: Dict[str, str]) -> Dict[str, Any]:
    """Builds the standard hydrated-context dict from a `chunks` table row."""
    doc_id = row["doc_id"]
    return {
        "chunk_id": row["chunk_id"],
        "raw_text": row["raw_text"],
        "tables_html": json.loads(row["tables_html"] or "[]"),
        "table_pages": json.loads(row["table_pages"] or "[]"),
        "images_base64": [],
        "image_pages": json.loads(row["image_pages"] or "[]"),
        "page_number": row["page_number"],
        "page_start": row["page_start"] if row["page_start"] is not None else row["page_number"],
        "page_end": row["page_end"] if row["page_end"] is not None else row["page_number"],
        "doc_id": doc_id,
        "filename": filenames.get(doc_id),
        "chunk_index": row["chunk_index"],
    }


async def perform_web_search(query: str) -> QueryResponse:
    global llm, web_search_tool
    logger.info(f"🔍 Performing WEB search for: {query}")

    if not query or not query.strip():
        # EDGE CASE: empty query string
        return QueryResponse(answer="Please enter a question.", citations=[], page_images=[])

    try:
        loop = asyncio.get_running_loop()
        search_results = await loop.run_in_executor(None, web_search_tool.results, query)

        if not search_results or "organic" not in search_results:
            logger.warning("No web results found.")
            return QueryResponse(
                answer="No web results found.",
                citations=[],
                page_images=[]
            )

        citations_for_frontend = []
        context_for_llm = ""

        for i, result in enumerate(search_results.get("organic", [])[:5]):
            snippet = result.get("snippet", "No snippet available.")
            title = result.get("title", "No title")
            url = result.get("link", "#")

            context_for_llm += f"--- Source {i + 1} ---\nTitle: {title}\nSnippet: {snippet}\nURL: {url}\n\n"

            citations_for_frontend.append(Citation(
                id=f"web_{uuid.uuid4()}",
                type="web",
                content=snippet,
                title=title,
                url=url
            ))

        prompt = f"""Based *only* on the following web search results, please answer this question: {query}

SEARCH RESULTS:
{context_for_llm}

Please provide a clear, comprehensive answer. If the results don't contain sufficient information, say "I couldn't find a clear answer in the web results."
If your answer includes tabular data, render it as a GitHub-flavored Markdown pipe
table rather than raw HTML.

ANSWER:"""

        response = await llm.ainvoke(prompt)
        answer = extract_text(response) or "I couldn't find a clear answer in the web results."

        return QueryResponse(
            answer=answer,
            citations=citations_for_frontend,
            page_images=[]
        )
    except Exception as e:
        logger.error(f"❌ Error during web search: {e}", exc_info=True)
        return QueryResponse(
            answer="Web search failed. Please try again.",
            citations=[],
            page_images=[]
        )


async def perform_document_search(request: QueryRequest) -> QueryResponse:
    logger.info(f"📄 Performing DOCUMENT search across doc_ids: {request.doc_ids}")

    if not request.query or not request.query.strip():
        # EDGE CASE: empty/whitespace-only query
        return QueryResponse(
            answer="Please enter a question.",
            citations=[],
            page_images=[]
        )

    if not request.doc_ids:
        return QueryResponse(
            answer="Please select at least one document.",
            citations=[],
            page_images=[]
        )

    conn = get_db()
    try:
        placeholders = ",".join("?" * len(request.doc_ids))
        doc_rows = conn.execute(
            f"SELECT doc_id, filename, status FROM documents WHERE doc_id IN ({placeholders})",
            request.doc_ids
        ).fetchall()
        filenames = {row["doc_id"]: row["filename"] for row in doc_rows}

        if not filenames:
            return QueryResponse(
                answer="The selected document(s) could not be found. Please re-upload.",
                citations=[],
                page_images=[]
            )

        # EDGE CASE: user selected a doc that is still processing or failed
        # ingestion. Tell them clearly instead of silently returning
        # "no relevant context found", which reads as a retrieval miss
        # rather than a data-availability problem.
        not_ready = [row["filename"] for row in doc_rows if row["status"] not in ("ready",)]
        if not_ready and len(not_ready) == len(doc_rows):
            statuses = {row["filename"]: row["status"] for row in doc_rows}
            still_processing = [f for f, s in statuses.items() if s == "processing"]
            failed = [f for f, s in statuses.items() if s == "failed"]
            if still_processing:
                return QueryResponse(
                    answer=f"'{', '.join(still_processing)}' is still being processed. Please try again shortly.",
                    citations=[], page_images=[]
                )
            if failed:
                return QueryResponse(
                    answer=f"'{', '.join(failed)}' failed to process and has no searchable content. Please re-upload it.",
                    citations=[], page_images=[]
                )

        # --- Whole-document intent (summaries, "list all X", "how many Y") ---
        # EDGE CASE: top-k similarity/MMR retrieval structurally cannot
        # answer "summarize this document" or "how many examples are there"
        # well, since it only ever returns a handful of locally-similar
        # chunks. Route these to a full-document map-reduce pass instead.
        if is_whole_document_query(request.query):
            logger.info("📚 Detected whole-document intent; using full-document summarization path.")
            answer = await generate_whole_document_summary(request.doc_ids, filenames, request.query)
            return QueryResponse(answer=answer, citations=[], page_images=[])

        # --- Direct page lookup ---
        # If the query names specific page(s) (e.g. "explain page 1",
        # "what's on page 6"), fetch those chunks directly by page number
        # instead of going through embedding-based retrieval. Vector search
        # has no concept of page numbers — it would just semantically match
        # the literal phrase "page 1" against whatever chunk talks about
        # page numbering, which is a different thing from the chunk that is
        # actually ON page 1. This path is exact and can't misfire.
        requested_pages = extract_requested_pages(request.query)
        if requested_pages:
            logger.info(f"📌 Detected explicit page request: {requested_pages}")
            all_rows = conn.execute(
                f"SELECT * FROM chunks WHERE doc_id IN ({placeholders}) ORDER BY doc_id, chunk_index",
                request.doc_ids
            ).fetchall()

            matched_contexts = []
            for row in all_rows:
                p_start = row["page_start"] if row["page_start"] is not None else row["page_number"]
                p_end = row["page_end"] if row["page_end"] is not None else row["page_number"]
                if p_start is None:
                    continue
                if p_end is None:
                    p_end = p_start
                if any(p_start <= p <= p_end for p in requested_pages):
                    matched_contexts.append(_hydrate_chunk_row(row, filenames))

            if not matched_contexts:
                # Tell the user the actual available page range instead of
                # silently falling through to a semantic search that would
                # likely retrieve the wrong content.
                range_row = conn.execute(
                    f"""SELECT MIN(COALESCE(page_start, page_number)) as min_pg,
                               MAX(COALESCE(page_end, page_number)) as max_pg
                        FROM chunks WHERE doc_id IN ({placeholders})""",
                    request.doc_ids
                ).fetchone()
                min_pg, max_pg = range_row["min_pg"], range_row["max_pg"]
                requested_label = ", ".join(str(p) for p in requested_pages)
                if min_pg is not None and max_pg is not None:
                    answer = (
                        f"Page(s) {requested_label} not found in the selected document(s). "
                        f"The selected document(s) cover pages {min_pg}-{max_pg}."
                    )
                else:
                    answer = f"Page(s) {requested_label} not found in the selected document(s)."
                return QueryResponse(answer=answer, citations=[], page_images=[])

            # Cap to avoid an oversized prompt if someone requests a huge range
            # (both by count and, inside generate_final_answer, by char budget).
            final_contexts = matched_contexts[:MAX_PAGE_MATCH_CHUNKS]

            citations_for_frontend = []
            page_images_for_frontend = []
            seen_page_images = set()
            for ctx in final_contexts:
                page_start = ctx.get("page_start")
                page_end = ctx.get("page_end") or page_start
                citations_for_frontend.append(Citation(
                    id=ctx["chunk_id"],
                    doc_id=ctx["doc_id"],
                    filename=ctx["filename"],
                    page=page_start,
                    page_end=page_end if page_end != page_start else None,
                    type="text",
                    content=((ctx["raw_text"] or "")[:200] + "...")
                ))
                doc_id = ctx["doc_id"]
                if page_start and page_end:
                    for pg in range(page_start, page_end + 1):
                        if (doc_id, pg) not in seen_page_images:
                            img = load_page_image(doc_id, pg)
                            if img:
                                page_images_for_frontend.append(PageImage(doc_id=doc_id, page=pg, image=img))
                                seen_page_images.add((doc_id, pg))

            answer = await generate_final_answer(final_contexts, request.query)
            return QueryResponse(
                answer=answer,
                citations=citations_for_frontend,
                page_images=page_images_for_frontend
            )

        # --- Stage 1: retrieve a wider candidate pool via MMR ---
        # MMR (Maximal Marginal Relevance) balances relevance with diversity,
        # so near-duplicate chunks don't crowd out topically distinct ones.
        # We deliberately over-fetch here (candidate_k) because stage 2
        # (reranking) does the real precision work — this stage's job is
        # just to make sure the right chunks are *somewhere* in the pool.
        candidate_k = min(6 * len(request.doc_ids), 30)
        final_k = min(4 * len(request.doc_ids), 10)

        retriever = pinecone_service.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": candidate_k,
                "fetch_k": min(candidate_k * 3, 60),
                "lambda_mult": 0.5,  # 0 = max diversity, 1 = pure relevance
                "filter": {"doc_id": {"$in": request.doc_ids}},
            }
        )
        loop = asyncio.get_running_loop()
        try:
            retrieved_chunks = await loop.run_in_executor(None, retriever.invoke, request.query)
        except Exception as e:
            # EDGE CASE: embedding/vector-store call itself can fail
            # (quota, network, dimension mismatch). Don't let this bubble
            # up as an opaque 500 for what looks like a normal question.
            logger.error(f"❌ Vector retrieval failed: {e}", exc_info=True)
            return QueryResponse(
                answer="Sorry, I ran into an error searching the selected documents. Please try again.",
                citations=[], page_images=[]
            )

        if not retrieved_chunks:
            logger.warning("No relevant chunks found in selected documents.")
            return QueryResponse(
                answer="Sorry, I couldn't find relevant context in the selected documents.",
                citations=[],
                page_images=[]
            )

        # Hydrate every candidate from SQLite (raw text, tables, page number, etc.)
        hydrated_candidates = []
        missing_chunk_ids = []
        for chunk_doc in retrieved_chunks:
            chunk_id = chunk_doc.metadata.get("chunk_id")
            doc_id = chunk_doc.metadata.get("doc_id")

            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if not row:
                missing_chunk_ids.append(chunk_id)
                continue

            hydrated_candidates.append({
                "chunk_id": chunk_id,
                "raw_text": row["raw_text"],
                "tables_html": json.loads(row["tables_html"] or "[]"),
                "table_pages": json.loads(row["table_pages"] or "[]"),
                "images_base64": [],
                "image_pages": json.loads(row["image_pages"] or "[]"),
                "page_number": row["page_number"],
                "page_start": row["page_start"] if row["page_start"] is not None else row["page_number"],
                "page_end": row["page_end"] if row["page_end"] is not None else row["page_number"],
                "doc_id": doc_id,
                "filename": filenames.get(doc_id),
            })

        if missing_chunk_ids:
            # EDGE CASE: Pinecone and SQLite have drifted out of sync (e.g. a
            # partial delete, or an interrupted upload that reached Pinecone
            # but not SQLite). This is a real data-consistency bug, distinct
            # from "no relevant content exists" — log it distinctly so it's
            # actionable rather than silently masked by a generic no-match answer.
            logger.warning(
                f"⚠️ {len(missing_chunk_ids)} chunk_id(s) returned by Pinecone were not found in SQLite "
                f"(possible index/DB drift): {missing_chunk_ids}"
            )

        if not hydrated_candidates:
            return QueryResponse(
                answer="Sorry, I couldn't find relevant context in the selected documents.",
                citations=[],
                page_images=[]
            )

        # --- Stage 2: rerank candidates against raw_text, keep the best final_k ---
        final_contexts = await rerank_chunks(request.query, hydrated_candidates, final_k)

        # Build citations and page images only from what actually goes to the LLM
        citations_for_frontend = []
        page_images_for_frontend = []
        seen_page_images = set()

        for ctx in final_contexts:
            page_start = ctx.get("page_start") or ctx.get("page_number")
            page_end = ctx.get("page_end") or page_start

            citations_for_frontend.append(Citation(
                id=ctx["chunk_id"],
                doc_id=ctx["doc_id"],
                filename=ctx["filename"],
                page=page_start,
                page_end=page_end if page_end != page_start else None,
                type="text",
                # Cite the raw source text, not the embedded AI summary
                content=((ctx["raw_text"] or "")[:200] + "...")
            ))

            doc_id = ctx["doc_id"]
            # Load page images for every page this chunk actually spans, not
            # just page_start — a table/image inside the chunk may live on a
            # later page than where the chunk's text begins.
            if page_start and page_end:
                for pg in range(page_start, page_end + 1):
                    if (doc_id, pg) not in seen_page_images:
                        img = load_page_image(doc_id, pg)
                        if img:
                            page_images_for_frontend.append(
                                PageImage(doc_id=doc_id, page=pg, image=img)
                            )
                            seen_page_images.add((doc_id, pg))

        answer = await generate_final_answer(final_contexts, request.query)

        return QueryResponse(
            answer=answer,
            citations=citations_for_frontend,
            page_images=page_images_for_frontend
        )
    except Exception as e:
        logger.error(f"❌ Error during document search: {e}", exc_info=True)
        return QueryResponse(
            answer="Document search failed. Please try again.",
            citations=[],
            page_images=[]
        )
    finally:
        conn.close()


async def _run_search_strategy(search_request: QueryRequest) -> QueryResponse:
    """
    Runs the document-only or hybrid (doc+web) strategy and returns a plain
    QueryResponse — with NO conversation_id/message_id attached yet. This
    is deliberately kept as a pure "given a query, find an answer +
    citations" function; conversation bookkeeping happens once, in the
    caller, so there's exactly one place where messages get persisted.
    """
    global llm

    # Strategy 1: Web Search is OFF. Only search selected documents.
    if not search_request.is_web_search_enabled:
        logger.info(f"Strategy: Document-Only for query: '{search_request.query}'")
        if not search_request.doc_ids:
            logger.warning("Document-Only query failed: no doc_ids provided")
            return QueryResponse(
                answer="Please select at least one document to ask questions, or enable the 'Web Search' toggle.",
                citations=[],
                page_images=[]
            )
        return await perform_document_search(search_request)

    # Strategy 2: Web Search is ON. Perform hybrid search (Doc + Web).
    logger.info(f"Strategy: Hybrid (Doc+Web) for query: '{search_request.query}'")

    search_tasks = [perform_web_search(search_request.query)]

    if search_request.doc_ids:
        logger.info("   → Hybrid: Adding document search to parallel tasks.")
        search_tasks.append(perform_document_search(search_request))
    else:
        logger.info("   → Hybrid: No documents selected, skipping document search.")

    logger.info(f"🚀 Running {len(search_tasks)} search tasks in parallel...")
    # EDGE CASE: gather with return_exceptions so a failure in one leg
    # (e.g. web search API outage) doesn't take down the other.
    results = await asyncio.gather(*search_tasks, return_exceptions=True)
    logger.info("✅ All search tasks complete.")

    def _safe_result(r, label):
        if isinstance(r, Exception):
            logger.error(f"❌ {label} search task raised: {r}", exc_info=True)
            return QueryResponse(answer="", citations=[], page_images=[])
        return r

    web_response = _safe_result(results[0], "Web")
    doc_response = _safe_result(results[1], "Document") if len(results) > 1 else None

    doc_answer = ""
    doc_citations = []
    doc_page_images = []

    if doc_response:
        doc_answer = doc_response.answer
        doc_citations = doc_response.citations
        doc_page_images = doc_response.page_images

    web_answer = web_response.answer
    web_citations = web_response.citations

    logger.info("   → Hybrid: Synthesizing answers...")

    if "Sorry, I couldn't find relevant context" in doc_answer:
        doc_answer = ""
    if "No web results found" in web_answer:
        web_answer = ""

    if not doc_answer and not web_answer:
        logger.warning("Hybrid search found no answer from any source.")
        final_answer = "Sorry, I couldn't find any information from your documents or the web."
    elif not doc_answer:
        logger.info("Hybrid search using Web-Only answer.")
        final_answer = web_answer
    elif not web_answer:
        logger.info("Hybrid search using Document-Only answer.")
        final_answer = doc_answer
    else:
        logger.info("Hybrid search synthesizing Doc and Web answers.")
        prompt = f"""You are a helpful assistant. You have received a user query and have two sources of information to answer it.

User Query: "{search_request.query}"

Source 1: Information from the uploaded document(s).
Document Answer:
{doc_answer}

Source 2: Information from a real-time web search.
Web Search Answer:
{web_answer}

Your Task:
Synthesize these two pieces of information into a single, comprehensive, and clear answer.
1. Prioritize the document information if it's available and relevant.
2. Use the web search to supplement, confirm, or answer parts the document couldn't.
3. If both sources say the same thing, just state the fact. Don't say "The document said... and the web said...".
4. If the document and web conflict, state the conflict clearly (e.g., "The document states X, but recent web results suggest Y.").
5. If only one source has an answer, use that.
6. If neither has an answer, state that you couldn't find the information.
7. If the final answer includes tabular data, render it as a GitHub-flavored
   Markdown pipe table rather than raw HTML.

Final Answer:
"""
        try:
            final_answer_response = await llm.ainvoke(prompt)
            final_answer = extract_text(final_answer_response) or "Error synthesizing results. Please try again."
        except Exception as e:
            logger.error(f"❌ Synthesizer LLM call failed: {e}")
            final_answer = "Error synthesizing results. Please try again."

    final_citations = doc_citations + web_citations

    return QueryResponse(
        answer=final_answer,
        citations=final_citations,
        page_images=doc_page_images
    )
