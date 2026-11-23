"""LLM quality gate. Runs the real pipeline and scores answers with Claude as judge.

    uv run pytest evals_python -m llm
"""

import json
import os
from pathlib import Path

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app import config, tracing
from app.rag import RagPipeline, claude_generator
from app.retriever import Retriever, load_chunks
from evals_python.judge import ClaudeJudge
from evals_python.scores import record_score

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text(encoding="utf-8"))
THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.7"))


@pytest.fixture(scope="module")
def pipeline():
    tracing.setup()
    return RagPipeline(Retriever(load_chunks(config.DOCS_DIR)), claude_generator())


@pytest.fixture(scope="module")
def metrics():
    judge = ClaudeJudge()
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Judge whether the actual output agrees with the expected output on the key facts. "
            "Penalize contradictions, missing key facts, and details that appear in neither the "
            "expected output nor the retrieval context. Extra details that are supported by the "
            "retrieval context are allowed and must not be penalized. If the expected output says "
            "the information is unavailable, the actual output must not invent an answer."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        threshold=THRESHOLD,
        model=judge,
    )
    return [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge),
        correctness,
    ]


@pytest.mark.parametrize("case", GOLDEN, ids=[c["input"][:40] for c in GOLDEN])
def test_rag_answer_quality(pipeline, metrics, case):
    result = pipeline.ask(case["input"])
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=result.answer,
        expected_output=case["expected_output"],
        retrieval_context=result.contexts or ["(no documents retrieved)"],
    )
    failures = []
    for metric in metrics:
        metric.measure(test_case)
        record_score(
            case=case["input"],
            metric=metric.__name__,
            score=metric.score,
            passed=metric.is_successful(),
            reason=metric.reason,
            framework="deepeval",
        )
        if not metric.is_successful():
            failures.append(f"{metric.__name__}: {metric.score:.2f} — {metric.reason}")
    assert not failures, "\n".join(failures)
