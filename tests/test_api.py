"""API tests: the agent is replaced by a fake, so no LLM and no models are needed."""
import pytest
from fastapi.testclient import TestClient

from wealthai import api

FAKE_RESULT = {
    "question": "How many clients do we have?",
    "answer": "There are 25 clients.",
    "answerable": True,
    "citations": [],
    "contexts": ["SQL: SELECT COUNT(*) FROM clients\n| n |\n|---|\n| 25 |"],
    "sources": ["portfolios.db"],
    "sections": ["sql result"],
    "mode": "multi",  # extra key: must not leak into the response schema
    "route": "portfolio",
    "sql": "SELECT COUNT(*) FROM clients",
    "trace": ["guard", "route -> portfolio"],
}


@pytest.fixture
def client():
    return TestClient(api.app, raise_server_exceptions=False)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask_returns_answer_and_evidence(client, monkeypatch):
    monkeypatch.setattr(api, "run_multi", lambda q: dict(FAKE_RESULT))
    r = client.post("/ask", json={"question": "How many clients do we have?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == "There are 25 clients." and body["route"] == "portfolio"
    assert body["contexts"] and body["trace"] and "mode" not in body


def test_question_is_stripped_before_reaching_the_agent(client, monkeypatch):
    seen = []
    monkeypatch.setattr(api, "run_multi", lambda q: seen.append(q) or dict(FAKE_RESULT))
    client.post("/ask", json={"question": "   How many clients do we have?  "})
    assert seen == ["How many clients do we have?"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "ab"}, {"question": "x" * 1001}, {"q": "hello there"}])
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_provider_rate_limit_maps_to_429(client, monkeypatch):
    class RateLimited(Exception):
        code = 429

    def boom(q):
        raise RateLimited("RESOURCE_EXHAUSTED secret-details")

    monkeypatch.setattr(api, "run_multi", boom)
    r = client.post("/ask", json={"question": "How many clients do we have?"})
    assert r.status_code == 429 and "secret-details" not in r.text


def test_unexpected_error_maps_to_503_without_leaking_details(client, monkeypatch):
    def boom(q):
        raise RuntimeError("api key sk-123 leaked in traceback")

    monkeypatch.setattr(api, "run_multi", boom)
    r = client.post("/ask", json={"question": "How many clients do we have?"})
    assert r.status_code == 503 and "sk-123" not in r.text
