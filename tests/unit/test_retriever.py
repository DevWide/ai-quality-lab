import pytest

from app import config
from app.retriever import Retriever, load_chunks


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks(config.DOCS_DIR))


@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("How much does the Team plan cost per user?", "plans-and-pricing.md"),
        # Regression: found by the DeepEval suite — SLA/refund chunks used to outrank pricing.
        ("How much does the Team plan cost if I pay annually?", "plans-and-pricing.md"),
        ("Can I get a refund on my annual subscription?", "refund-policy.md"),
        ("Is two-factor authentication mandatory?", "security.md"),
        ("What is the uptime SLA for Enterprise?", "support.md"),
    ],
)
def test_top_result_comes_from_expected_document(retriever, question, expected_source):
    results = retriever.search(question, k=3)
    assert results, "retriever returned nothing"
    assert results[0].source == expected_source


def test_unrelated_question_returns_nothing(retriever):
    assert retriever.search("zzz qqq xyzzy") == []


def test_respects_k(retriever):
    assert len(retriever.search("plan", k=2)) <= 2


def test_scored_results_are_positive_and_ranked(retriever):
    scores = [score for _, score in retriever.search_scored("refund annual plan", k=3)]
    assert scores and all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)
