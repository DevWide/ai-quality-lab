"""Keyword retrieval with BM25.

BM25 is deterministic and needs no embeddings API, which keeps CI cheap and
reproducible. Swapping it for vector search later is a good experiment to
measure with the evals.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    # Minimal plural folding so "costs" matches "cost"; not a real stemmer.
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


def _split_paragraph(paragraph: str) -> list[str]:
    """Bullet lists become one chunk per item, so one long list doesn't dilute
    every item's BM25 score. The lead-in line ("Support channels by plan:")
    is kept on each item for context."""
    lines = paragraph.strip().splitlines()
    items = [line for line in lines if line.startswith("- ")]
    if not items:
        return [paragraph.strip()]
    lead = "\n".join(line for line in lines if not line.startswith("- "))
    return [f"{lead}\n{item}".strip() for item in items]


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Paragraph (or list item) chunks, prefixed with the document title."""
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
        title = blocks[0].lstrip("# ").strip()
        for block in blocks[1:]:
            for text in _split_paragraph(block):
                chunks.append(Chunk(source=path.name, text=f"{title}\n{text}"))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No documents to index")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        return [chunk for chunk, _ in self.search_scored(query, k)]

    def search_scored(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]
