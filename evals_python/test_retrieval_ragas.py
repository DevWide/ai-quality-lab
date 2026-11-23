"""Retrieval quality with Ragas: did the search bring the right chunks?

DeepEval (test_rag_quality.py) mostly grades the answer. These metrics grade the
retrieval step, so a bad answer can be traced to search vs. generation:

- context_precision: are the retrieved chunks relevant, with the useful ones ranked first?
- context_recall: does the retrieved text contain everything the reference answer needs?
- faithfulness: is every claim in the answer supported by the retrieved text?
  (also measured by DeepEval; kept here to compare two judges on the same cases)

    uv run --group ragas pytest evals_python/test_retrieval_ragas.py -m llm -v
"""

import asyncio
import json
import math
import os
from pathlib import Path

import pytest

pytest.importorskip("ragas", reason="install with: uv sync --group ragas")

from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness  # noqa: E402

from app import config  # noqa: E402
from app.rag import RagPipeline, claude_generator  # noqa: E402
from app.retriever import Retriever, load_chunks  # noqa: E402
from evals_python.ragas_judge import ClaudeRagasJudge  # noqa: E402
from evals_python.scores import record_score  # noqa: E402

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text(encoding="utf-8"))
# Retrieval metrics need a reference answer that comes from the docs; unanswerable cases have none.
ANSWERABLE = [c for c in GOLDEN if c.get("answerable", True)]
THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.7"))


@pytest.fixture(scope="module")
def pipeline():
    return RagPipeline(Retriever(load_chunks(config.DOCS_DIR)), claude_generator())


@pytest.fixture(scope="module")
def judge():
    return ClaudeRagasJudge()


async def score_case(judge, question: str, reference: str, answer: str, contexts: list[str]) -> dict:
    # Ragas' faithfulness checks the answer against the retrieved text only, not the question.
    # So when the bot restates a fact the user gave ("20 days have passed"), the judge marks it
    # as made up and every conclusion built on it fails too. Facts from the user are fair to use,
    # so the question is passed to that metric as one more piece of context.
    question_as_context = f"The user's question (facts stated here were given by the user): {question}"
    precision, recall, faithfulness = await asyncio.gather(
        ContextPrecision(llm=judge).ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        ContextRecall(llm=judge).ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        Faithfulness(llm=judge).ascore(
            user_input=question, response=answer, retrieved_contexts=[*contexts, question_as_context]
        ),
    )
    return {
        "context_precision": precision.value,
        "context_recall": recall.value,
        "ragas_faithfulness": faithfulness.value,
    }


@pytest.mark.parametrize("case", ANSWERABLE, ids=[c["input"][:40] for c in ANSWERABLE])
def test_retrieval_quality(pipeline, judge, case):
    result = pipeline.ask(case["input"])
    assert result.contexts, "retriever returned nothing for an answerable question"

    scores = asyncio.run(score_case(judge, case["input"], case["expected_output"], result.answer, result.contexts))

    failures = []
    for name, score in scores.items():
        passed = score is not None and not math.isnan(score) and score >= THRESHOLD
        record_score(case=case["input"], metric=name, score=score, passed=passed, reason=None, framework="ragas")
        if not passed:
            failures.append(f"{name}: {score}")
    assert not failures, "\n".join(failures) + "\nsources: " + ", ".join(result.sources)
