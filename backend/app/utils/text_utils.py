import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def _truncate_contexts_to_budget(chunks: List[Dict], max_chars: int) -> List[Dict]:
    """
    EDGE CASE: capping by chunk *count* (e.g. top 10) doesn't bound prompt
    size, since chunks vary wildly in length (a table-heavy chunk can be
    many times larger than a text-only one). This caps by a character/token
    budget instead, always including at least one chunk so the model has
    *something* to work with even if it's oversized.
    """
    if not chunks:
        return chunks

    budgeted = []
    running = 0
    for ctx in chunks:
        size = len(ctx.get("raw_text") or "")
        for table in ctx.get("tables_html", []) or []:
            size += len(table or "")
        if budgeted and running + size > max_chars:
            break
        budgeted.append(ctx)
        running += size

    return budgeted or chunks[:1]


def _extract_cited_pages(answer: str) -> List[str]:
    """Pulls out bracketed citations like [Doc, Page 5] or [Doc, Pages 5-6] from an answer."""
    return re.findall(r"\[[^\[\]]*?[Pp]ages?\s*[\d\-–toTO ]+\]", answer)


def _validate_citations(answer: str, chunks: List[Dict]) -> None:
    """
    EDGE CASE: LLMs can occasionally cite a page number that wasn't
    actually part of the supplied context. This is a soft, log-only check
    (not a hard block) — flags likely-hallucinated citations for
    observability without risking false positives blocking a good answer.
    """
    cited = _extract_cited_pages(answer)
    if not cited:
        return
    valid_pages = set()
    for ctx in chunks:
        p_start = ctx.get("page_start") or ctx.get("page_number")
        p_end = ctx.get("page_end") or p_start
        if p_start and p_end:
            valid_pages.update(range(int(p_start), int(p_end) + 1))
    for citation in cited:
        nums = [int(n) for n in re.findall(r"\d+", citation)]
        if nums and not any(n in valid_pages for n in nums):
            logger.warning(f"⚠️ Possible hallucinated citation not present in retrieved context: {citation}")
