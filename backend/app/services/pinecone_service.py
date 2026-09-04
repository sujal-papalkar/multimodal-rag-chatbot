import os
import sys
import asyncio
import logging
from typing import List, Dict
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSIONS,
    RERANK_MODEL,
    GEMINI_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# --- Global Inits ---
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBEDDING_MODEL,
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    output_dimensionality=EMBEDDING_DIMENSIONS,
)
vectorstore = None
pinecone_index = None  # raw pinecone.Index handle, used for ID-based deletes and reranking


def init_pinecone():
    global vectorstore, pinecone_index

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
            )
            logger.info("✅ Index created.")
        except Exception as e:
            logger.error(f"❌ Pinecone index creation failed: {e}")
            raise
    else:
        logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Sync module globals across any app modules that imported vectorstore/pinecone_index directly
    for mod_name, mod in list(sys.modules.items()):
        if mod and mod_name.startswith("app."):
            if hasattr(mod, "vectorstore"):
                setattr(mod, "vectorstore", vectorstore)
            if hasattr(mod, "pinecone_index"):
                setattr(mod, "pinecone_index", pinecone_index)

    logger.info("✅ Pinecone vector store initialized.")


async def rerank_chunks(query: str, hydrated_contexts: List[Dict], top_n: int) -> List[Dict]:
    """
    Re-scores retrieved chunks against the query using Pinecone's hosted
    cross-encoder reranker (bge-reranker-v2-m3), then returns the top_n
    contexts in reranked order.

    This is a second-stage refinement on top of first-stage vector
    similarity: similarity/MMR search is fast but approximate (and here,
    it's matching against an LLM-generated summary, not the raw text), so
    a wider candidate pool is retrieved first and then re-scored against
    the actual raw_text for more accurate final ranking.

    Falls back to the original (similarity/MMR) order, truncated to
    top_n, if reranking fails for any reason (e.g. rerank model
    unavailable, quota, network issue) so a rerank outage never breaks
    the whole query.
    """
    if not hydrated_contexts:
        return hydrated_contexts

    # Rerank against raw_text (the actual source content) rather than the
    # AI summary that was embedded, since raw_text is more faithful to
    # what's really in the document.
    documents = [
        {"id": str(i), "text": (ctx.get("raw_text") or "")[:2000] or " "}
        for i, ctx in enumerate(hydrated_contexts)
    ]

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pc.inference.rerank(
                model=RERANK_MODEL,
                query=query,
                documents=documents,
                rank_fields=["text"],
                top_n=min(top_n, len(documents)),
                return_documents=False,
            )
        )
        reranked = [hydrated_contexts[item.index] for item in result.data]
        if reranked:
            logger.info(f"🎯 Reranked {len(hydrated_contexts)} candidates -> top {len(reranked)}")
            return reranked
        logger.warning("Rerank returned no results, falling back to similarity order")
    except Exception as e:
        logger.warning(f"⚠️ Reranking failed, falling back to similarity/MMR order: {e}")

    return hydrated_contexts[:top_n]
