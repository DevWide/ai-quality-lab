from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app import config, tracing
from app.rag import RagPipeline, claude_generator
from app.retriever import Retriever, load_chunks

tracing.setup()
app = FastAPI(title="Orbita Cloud RAG assistant")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str]


@lru_cache
def get_pipeline() -> RagPipeline:
    return RagPipeline(Retriever(load_chunks(config.DOCS_DIR)), claude_generator())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> AskResponse:
    result = pipeline.ask(req.question)
    return AskResponse(answer=result.answer, sources=result.sources, contexts=result.contexts)
