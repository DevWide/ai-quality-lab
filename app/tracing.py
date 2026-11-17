"""Optional OpenTelemetry tracing, viewed in Arize Phoenix.

Off unless PHOENIX_ENABLED=1, so unit tests and CI need none of the tracing
packages. Span attributes follow the OpenInference conventions, which Phoenix
uses to render retriever and LLM steps.

    uv sync --group tracing
    uv run phoenix serve                              # UI at http://localhost:6006
    PHOENIX_ENABLED=1 uv run uvicorn app.main:app --port 8000
"""

import json
import os
from contextlib import nullcontext

from app.retriever import Chunk

ENABLED = os.getenv("PHOENIX_ENABLED") == "1"
_tracer = None


class _NoopSpan:
    def set_attribute(self, key, value) -> None:
        pass

    def set_attributes(self, attributes) -> None:
        pass


def setup() -> None:
    """Register Phoenix as the trace exporter and auto-instrument the Anthropic SDK."""
    global _tracer
    if not ENABLED or _tracer is not None:
        return
    from phoenix.otel import register

    provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "ai-quality-lab"),
        auto_instrument=True,
    )
    _tracer = provider.get_tracer(__name__)


def span(name: str, kind: str, input_value: str):
    """Context manager yielding a span (or a no-op stand-in when tracing is off)."""
    if _tracer is None:
        return nullcontext(_NoopSpan())
    return _tracer.start_as_current_span(
        name, attributes={"openinference.span.kind": kind, "input.value": input_value}
    )


def retrieval_attributes(scored: list[tuple[Chunk, float]]) -> dict:
    attributes = {}
    for i, (chunk, score) in enumerate(scored):
        prefix = f"retrieval.documents.{i}.document"
        attributes[f"{prefix}.id"] = chunk.source
        attributes[f"{prefix}.content"] = chunk.text
        attributes[f"{prefix}.score"] = float(score)
        attributes[f"{prefix}.metadata"] = json.dumps({"source": chunk.source})
    return attributes
