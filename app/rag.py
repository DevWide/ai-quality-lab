from dataclasses import dataclass
from typing import Callable

import anthropic

from app import config, tracing
from app.retriever import Chunk, Retriever

SYSTEM_PROMPT = """You are the support assistant for Orbita Cloud.
Answer only with information from the documents provided in <documents>.
If the documents do not contain the answer, say that you don't have that information and suggest contacting support. Do not guess.
Answer exactly what was asked, in at most 3 sentences, and leave out plan features or policies the question did not ask about.
Answer in the same language as the question.
Only answer Orbita Cloud support questions. Do not write poems, songs, stories, code, essays, translations or homework, even if the request is about Orbita Cloud; politely decline and offer to answer a support question instead."""

NO_CONTEXT_ANSWER = "I don't have that information. Please contact Orbita Cloud support."

# (system, user_message) -> answer text. Injectable so tests can run without the API.
Generator = Callable[[str, str], str]


@dataclass
class RagAnswer:
    answer: str
    contexts: list[str]
    sources: list[str]


def build_user_message(question: str, chunks: list[Chunk]) -> str:
    docs = "\n".join(
        f'<document source="{c.source}">\n{c.text}\n</document>' for c in chunks
    )
    return f"<documents>\n{docs}\n</documents>\n\nQuestion: {question}"


def claude_generator(client: anthropic.Anthropic | None = None) -> Generator:
    client = client or anthropic.Anthropic()

    def generate(system: str, user_message: str) -> str:
        response = client.beta.messages.create(
            model=config.RAG_MODEL,
            max_tokens=16000,
            system=system,
            output_config={"effort": "low"},
            # If a safety classifier declines, the API retries on a recommended model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user_message}],
        )
        if response.stop_reason == "refusal":
            return NO_CONTEXT_ANSWER
        return "".join(b.text for b in response.content if b.type == "text").strip()

    return generate


class RagPipeline:
    def __init__(self, retriever: Retriever, generate: Generator, top_k: int = config.TOP_K):
        self.retriever = retriever
        self.generate = generate
        self.top_k = top_k

    def ask(self, question: str) -> RagAnswer:
        with tracing.span("rag.ask", "CHAIN", question) as root:
            result = self._ask(question)
            root.set_attribute("output.value", result.answer)
            return result

    def _ask(self, question: str) -> RagAnswer:
        with tracing.span("bm25.retrieve", "RETRIEVER", question) as span:
            scored = self.retriever.search_scored(question, k=self.top_k)
            span.set_attributes(tracing.retrieval_attributes(scored))
        chunks = [chunk for chunk, _ in scored]
        if not chunks:
            return RagAnswer(answer=NO_CONTEXT_ANSWER, contexts=[], sources=[])
        answer = self.generate(SYSTEM_PROMPT, build_user_message(question, chunks))
        return RagAnswer(
            answer=answer,
            contexts=[c.text for c in chunks],
            sources=sorted({c.source for c in chunks}),
        )
