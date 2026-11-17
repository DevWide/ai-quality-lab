import json

from app import tracing
from app.retriever import Chunk


def test_span_is_noop_when_tracing_is_off():
    # Unit tests run without PHOENIX_ENABLED, so no tracer is registered.
    with tracing.span("rag.ask", "CHAIN", "question") as span:
        span.set_attribute("output.value", "answer")
        span.set_attributes({"a": 1})


def test_retrieval_attributes_follow_openinference_keys():
    attrs = tracing.retrieval_attributes([(Chunk(source="support.md", text="SLA text"), 2.5)])
    assert attrs == {
        "retrieval.documents.0.document.id": "support.md",
        "retrieval.documents.0.document.content": "SLA text",
        "retrieval.documents.0.document.score": 2.5,
        "retrieval.documents.0.document.metadata": json.dumps({"source": "support.md"}),
    }
