"""Coverage of cross-fund questions: a ranking over the library must see every fund. No models, no index."""
import pytest

from wealthai import retrieval

KEYS = {
    "a.pdf": {"ishares", "msci", "world"},
    "b.pdf": {"ishares", "s&p", "500"},
    "c.pdf": {"vanguard", "ftse", "all-world"},
    "d.pdf": {"xtrackers", "msci", "world"},
    "e.pdf": {"ishares", "msci", "em", "imi"},
}


@pytest.fixture(autouse=True)
def fake_funds(monkeypatch):
    monkeypatch.setattr(retrieval, "fund_keys", lambda: KEYS)


@pytest.mark.parametrize("question", [
    "Which fund has the lowest total costs in USD if you exit after 1 year?",
    "Which of the funds has the highest ongoing costs?",
    "What is the cheapest fund over five years?",
    "How many funds have a risk indicator of 4?",
    "Which fund's depositary is Brown Brothers Harriman?",
    "Compare the funds",
])
def test_unnamed_ranking_questions_are_cross_fund(question):
    assert retrieval.is_cross_fund(question)


@pytest.mark.parametrize("question", [
    "What are the ongoing costs of the Vanguard FTSE All-World fund?",
    "Is the iShares S&P 500 cheaper than the Xtrackers MSCI World?",  # names funds: per-fund path already applies
    "What is the weather in Nyon?",
])
def test_named_or_unrelated_questions_are_not_cross_fund(question):
    assert not retrieval.is_cross_fund(question)


def test_all_funds_gives_every_fund_its_own_share(monkeypatch):
    searched = []

    def fake_search(query, n, sources=None):
        searched.append(tuple(sources or ()))
        src = sources[0] if sources and len(sources) == 1 else "a.pdf"
        return [{"chunk_id": f"{src}-{i}", "source": src, "text": f"{src} {i}", "score": 1.0} for i in range(3)]

    monkeypatch.setattr(retrieval, "dense_search", fake_search)
    monkeypatch.setattr(retrieval, "bm25_search", fake_search)
    monkeypatch.setattr(retrieval, "rerank", lambda q, cands, k: [dict(c, score=1.0) for c in cands[:k]])
    out = retrieval.retrieve("total costs", k=5, mode="hybrid_perfund_rerank", all_funds=True)
    assert {c["source"] for c in out} == set(KEYS)
    assert all(sum(c["source"] == s for c in out) == 2 for s in KEYS)  # at least 2 chunks per fund


def test_all_funds_is_ignored_by_modes_without_per_fund_balancing(monkeypatch):
    calls = []
    monkeypatch.setattr(retrieval, "dense_search", lambda q, n, sources=None: calls.append(sources) or [])
    monkeypatch.setattr(retrieval, "bm25_search", lambda q, n, sources=None: [])
    monkeypatch.setattr(retrieval, "rerank", lambda q, c, k: [])
    retrieval.retrieve("total costs", k=5, mode="hybrid_rerank", all_funds=True)
    assert calls == [None]


def test_a_widened_retry_gets_more_chunks_per_fund(monkeypatch):
    def fake_search(query, n, sources=None):
        src = sources[0] if sources and len(sources) == 1 else "a.pdf"
        return [{"chunk_id": f"{src}-{i}", "source": src, "text": f"{src} {i}", "score": 1.0} for i in range(6)]

    monkeypatch.setattr(retrieval, "dense_search", fake_search)
    monkeypatch.setattr(retrieval, "bm25_search", fake_search)
    monkeypatch.setattr(retrieval, "rerank", lambda q, cands, k: [dict(c, score=1.0) for c in cands[:k]])
    first = retrieval.retrieve("total costs", k=5, mode="hybrid_perfund_rerank", all_funds=True)
    retry = retrieval.retrieve("total costs", k=8, mode="hybrid_perfund_rerank", all_funds=True)  # the agent widens k on a retry
    assert len(retry) > len(first)  # before the fix both returned 2 chunks per fund
    assert all(sum(c["source"] == s for c in retry) == 3 for s in KEYS)
