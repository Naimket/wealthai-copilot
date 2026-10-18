"""Headless UI test with Streamlit's AppTest. Skipped when Streamlit is not installed (it lives in the `ui` group)."""
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
RESULT = {
    "question": "q", "answer": "There are 25 clients.", "answerable": True, "route": "portfolio",
    "sql": "SELECT COUNT(*) FROM clients", "citations": [], "sources": ["portfolios.db"],
    "sections": ["sql result"], "contexts": ["| n |\n|---|\n| 25 |"], "trace": ["guard", "route -> portfolio"],
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code, self._payload = status_code, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=httpx.Request("POST", "http://x"), response=self)


def run_app(monkeypatch, response=None, error=None):
    def fake_post(url, json=None, timeout=None):
        if error:
            raise error
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.text_area[0].set_value("How many clients do we have?").run()
    ask = next(b for b in at.button if b.label == "Ask")
    ask.click().run()
    return at


def test_answer_and_evidence_are_shown(monkeypatch):
    at = run_app(monkeypatch, FakeResponse(200, RESULT))
    assert not at.exception
    assert at.success[0].value == "There are 25 clients."
    labels = [e.label for e in at.expander]
    assert any("SQL" in label for label in labels) and any("Evidence" in label for label in labels)


def test_unanswerable_is_a_warning_not_a_success(monkeypatch):
    at = run_app(monkeypatch, FakeResponse(200, {**RESULT, "answerable": False, "answer": "Not enough information."}))
    assert at.warning[0].value == "Not enough information." and not at.success


def test_api_error_detail_is_shown(monkeypatch):
    at = run_app(monkeypatch, FakeResponse(429, {"detail": "Rate limit reached."}))
    assert at.error[0].value == "Rate limit reached."


def test_unreachable_api_gives_a_clear_message(monkeypatch):
    at = run_app(monkeypatch, error=httpx.ConnectError("refused"))
    assert "Could not reach the API" in at.error[0].value


def test_ask_button_is_disabled_for_empty_question(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("API must not be called"))
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert next(b for b in at.button if b.label == "Ask").disabled
