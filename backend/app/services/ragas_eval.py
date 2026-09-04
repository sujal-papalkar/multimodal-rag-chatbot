"""
RAGAS-based evaluation for the production RAG pipeline.

Metrics:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall (when ground_truth is supplied)

Usage:
    POST /evaluate

Example:
{
  "cases": [
    {
      "question": "What is the value of gravitational acceleration at the surface of the earth?",
      "doc_ids": ["<doc-id>"],
      "ground_truth": "The value of g at the surface of the earth is approximately 9.8 m/s^2."
    }
  ]
}

Standalone:
    python -m app.services.ragas_eval eval_set.json
"""
print("🔥 NEW RAGAS_EVAL.PY IS RUNNING 🔥")
import asyncio
import json
import logging
import sys
import types
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vertex AI compatibility shim
#
# RAGAS 0.4.3 may try to import:
#     langchain_community.chat_models.vertexai
#
# Newer LangChain installations may expose ChatVertexAI through:
#     langchain_google_vertexai
#
# Create the compatibility module before importing RAGAS.
# ---------------------------------------------------------------------------

try:
    import langchain_google_vertexai

    vertex_module_name = "langchain_community.chat_models.vertexai"

    if vertex_module_name not in sys.modules:
        vertex_module = types.ModuleType(vertex_module_name)
        vertex_module.ChatVertexAI = getattr(
            langchain_google_vertexai,
            "ChatVertexAI",
            None,
        )
        sys.modules[vertex_module_name] = vertex_module

except ImportError:
    logger.warning(
        "langchain_google_vertexai is not installed. "
        "RAGAS VertexAI compatibility shim was not created."
    )


# ---------------------------------------------------------------------------
# RAGAS imports
# ---------------------------------------------------------------------------

try:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    RAGAS_AVAILABLE = True

    try:
        import ragas

        RAGAS_VERSION = getattr(ragas, "__version__", "unknown")
    except Exception:
        RAGAS_VERSION = "unknown"

    logger.info("RAGAS loaded successfully. Version: %s", RAGAS_VERSION)

except ImportError as exc:
    RAGAS_AVAILABLE = False
    RAGAS_VERSION = "unavailable"

    logger.exception(
        "RAGAS could not be imported. "
        "Run `pip install ragas` to enable /evaluate. "
        "Import error: %s",
        exc,
    )


# ===========================================================================
# Request / response models
# ===========================================================================


class EvalCase(BaseModel):
    question: str
    doc_ids: List[str]

    # Optional reference answer.
    # ContextRecall requires a reference answer.
    ground_truth: Optional[str] = None


class EvalRequest(BaseModel):
    cases: List[EvalCase]


class EvalCaseResult(BaseModel):
    question: str
    answer: str

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class EvalResponse(BaseModel):
    results: List[EvalCaseResult]
    averages: Dict[str, float]


# ===========================================================================
# Core evaluation
# ===========================================================================


async def run_ragas_evaluation(
    cases: List[EvalCase],
    search_fn,
    llm,
    embeddings,
) -> EvalResponse:
    """
    Run RAGAS evaluation using the application's actual retrieval pipeline.

    Parameters
    ----------
    cases:
        Evaluation questions and document IDs.

    search_fn:
        The application's real document search function.

    llm:
        Same LLM used by the production RAG pipeline.

    embeddings:
        Same embedding model used by the application.
    """

    if not RAGAS_AVAILABLE:
        raise RuntimeError(
            "RAGAS could not be imported. "
            "Run `pip install ragas` and check the backend dependencies."
        )

    # Import lazily to avoid circular imports.
    from app.models.schemas import QueryRequest

    samples = []
    raw_answers = []

    # -----------------------------------------------------------------------
    # Run production retrieval + generation for every test case
    # -----------------------------------------------------------------------

    for case_index, case in enumerate(cases):

        logger.info(
            "Running evaluation case %d/%d: %s",
            case_index + 1,
            len(cases),
            case.question,
        )

        request = QueryRequest(
            query=case.question,
            doc_ids=case.doc_ids,
            is_web_search_enabled=False,
        )

        # IMPORTANT:
        # This is the SAME search function used by the actual application.
        response = await search_fn(request)

        # -------------------------------------------------------------------
        # Extract retrieved contexts from citations
        # -------------------------------------------------------------------

        contexts = []

        if response.citations:
            for citation in response.citations:
                content = getattr(citation, "content", None)

                if content:
                    contexts.append(content)

        # -------------------------------------------------------------------
        # DEBUG INFORMATION
        # -------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("RAGAS DEBUG")
        print("=" * 80)

        print("Case:", case_index + 1)
        print("Question:")
        print(case.question)

        print("\nDocument IDs:")
        print(case.doc_ids)

        print("\nAnswer:")
        print(response.answer)

        print("\nNumber of citations:")
        print(len(response.citations))

        print("\nNumber of contexts:")
        print(len(contexts))

        for i, context in enumerate(contexts):
            print("\n" + "-" * 80)
            print(f"CONTEXT {i + 1}")
            print("-" * 80)

            # Print enough text to inspect retrieval,
            # but don't flood the terminal with huge chunks.
            print(context[:3000])

        print("\n" + "=" * 80)
        print("END RAGAS DEBUG")
        print("=" * 80 + "\n")

        # -------------------------------------------------------------------
        # Empty retrieval handling
        # -------------------------------------------------------------------

        if not contexts:
            logger.warning(
                "No contexts were retrieved for question: %s",
                case.question,
            )

            contexts = [
                "[no context retrieved for this question]"
            ]

        # -------------------------------------------------------------------
        # Build RAGAS sample
        # -------------------------------------------------------------------

        sample_kwargs = {
            "user_input": case.question,
            "response": response.answer,
            "retrieved_contexts": contexts,
        }

        # Only provide reference when supplied.
        if case.ground_truth:
            sample_kwargs["reference"] = case.ground_truth

        samples.append(
            SingleTurnSample(**sample_kwargs)
        )

        raw_answers.append(response.answer)

    # =========================================================================
    # Build evaluation dataset
    # =========================================================================

    dataset = EvaluationDataset(
        samples=samples
    )

    # =========================================================================
    # Wrap production LLM and embeddings for RAGAS
    # =========================================================================

    ragas_llm = LangchainLLMWrapper(llm)

    ragas_embeddings = LangchainEmbeddingsWrapper(
        embeddings
    )

    # =========================================================================
    # Metrics
    # =========================================================================

    metrics = [
        Faithfulness(
            llm=ragas_llm
        ),

        AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        ),

        ContextPrecision(
            llm=ragas_llm,
        ),
    ]

    # Context recall requires reference/ground truth.
    has_ground_truth = any(
        case.ground_truth
        for case in cases
    )

    if has_ground_truth:
        metrics.append(
            ContextRecall(
                llm=ragas_llm
            )
        )

    logger.info(
        "Evaluating %d cases using %d metrics",
        len(samples),
        len(metrics),
    )

    # =========================================================================
    # Run RAGAS
    # =========================================================================

    loop = asyncio.get_running_loop()

    def _run_evaluation():
        return evaluate(
            dataset=dataset,
            metrics=metrics,
        )

    result = await loop.run_in_executor(
        None,
        _run_evaluation,
    )

    # =========================================================================
    # Convert result to pandas
    # =========================================================================

    scored = result.to_pandas()

    logger.info(
        "RAGAS evaluation completed successfully."
    )

    # =========================================================================
    # Build API response
    # =========================================================================

    results: List[EvalCaseResult] = []

    for i, case in enumerate(cases):

        row = scored.iloc[i]

        results.append(
            EvalCaseResult(
                question=case.question,
                answer=raw_answers[i],

                faithfulness=_safe_float(
                    row.get("faithfulness")
                ),

                answer_relevancy=_safe_float(
                    row.get("answer_relevancy")
                ),

                context_precision=_safe_float(
                    row.get("context_precision")
                ),

                context_recall=(
                    _safe_float(
                        row.get("context_recall")
                    )
                    if "context_recall" in row
                    else None
                ),
            )
        )

    # =========================================================================
    # Calculate averages
    # =========================================================================

    averages: Dict[str, float] = {}

    for field in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):

        values = [
            getattr(result, field)
            for result in results
            if getattr(result, field) is not None
        ]

        if values:
            averages[field] = sum(values) / len(values)

    return EvalResponse(
        results=results,
        averages=averages,
    )


# ===========================================================================
# Safe numeric conversion
# ===========================================================================


def _safe_float(value: Any) -> Optional[float]:
    """
    Safely convert a RAGAS score to float.

    Handles:
      - None
      - NaN
      - strings
      - unexpected values
    """

    try:

        if value is None:
            return None

        f = float(value)

        # NaN check
        if f != f:
            return None

        return f

    except (
        TypeError,
        ValueError,
    ):
        return None


# ===========================================================================
# Standalone CLI runner
# ===========================================================================


if __name__ == "__main__":

    async def _main():

        if len(sys.argv) < 2:

            print(
                "Usage: "
                "python -m app.services.ragas_eval eval_set.json"
            )

            print(
                'Example: '
                '[{"question": "...", '
                '"doc_ids": ["..."], '
                '"ground_truth": "..."}]'
            )

            sys.exit(1)

        # -------------------------------------------------------------------
        # Load evaluation dataset
        # -------------------------------------------------------------------

        with open(
            sys.argv[1],
            encoding="utf-8",
        ) as f:

            raw_cases = json.load(f)

        cases = [
            EvalCase(**case)
            for case in raw_cases
        ]

        # -------------------------------------------------------------------
        # Import application services
        # -------------------------------------------------------------------

        from app.services.pinecone_service import (
            init_pinecone,
            embeddings,
        )

        from app.services.llm_service import llm

        from app.services.search_service import (
            perform_document_search,
        )

        from app.db.database import init_db

        # -------------------------------------------------------------------
        # Initialize application services
        # -------------------------------------------------------------------

        init_pinecone()
        init_db()

        # -------------------------------------------------------------------
        # Run evaluation
        # -------------------------------------------------------------------

        response = await run_ragas_evaluation(
            cases=cases,
            search_fn=perform_document_search,
            llm=llm,
            embeddings=embeddings,
        )

        print(
            json.dumps(
                response.model_dump(),
                indent=2,
            )
        )

    asyncio.run(_main())