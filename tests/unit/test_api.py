from fastapi.testclient import TestClient

from app import config
from app.main import app, get_pipeline
from app.rag import NO_CONTEXT_ANSWER, RagPipeline
from app.retriever import Retriever, load_chunks


class FakeGenerator:
    """Records what the pipeline sends to the LLM and returns a canned answer."""

    def __init__(self):
        self.calls = []

    def __call__(self, system: str, user_message: str) -> str:
        self.calls.append((system, user_message))
        return "Fake answer"


def make_client(generator):
    pipeline = RagPipeline(Retriever(load_chunks(config.DOCS_DIR)), generator)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


def test_ask_sends_retrieved_documents_to_llm():
    fake = FakeGenerator()
    response = make_client(fake).post("/ask", json={"question": "How long are backups kept?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Fake answer"
    assert "security.md" in body["sources"]
    _, user_message = fake.calls[0]
    assert "35 days" in user_message
    assert "How long are backups kept?" in user_message


def test_no_matching_documents_skips_llm():
    fake = FakeGenerator()
    response = make_client(fake).post("/ask", json={"question": "xyzzy qqq"})

    assert response.json()["answer"] == NO_CONTEXT_ANSWER
    assert fake.calls == []


def test_empty_question_is_rejected():
    response = make_client(FakeGenerator()).post("/ask", json={"question": ""})
    assert response.status_code == 422


def teardown_module():
    app.dependency_overrides.clear()
