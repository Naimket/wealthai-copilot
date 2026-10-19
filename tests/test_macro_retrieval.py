"""Macro retrieval wiring with fake chunks and fake dense/rerank steps: no index, no models."""
import pytest

from wealthai import retrieval_macro as rm

CHUNKS = [
    {"chunk_id": "a", "text": "[Bulletin | Prices | PDF p.20]\nAnnual CPI inflation stood at 0.8% in August.", "source": "b.pdf"},
    {"chunk_id": "b", "text": "[Bulletin | Markets | PDF p.26]\nThe Swiss franc lost 3.8% against the US dollar.", "source": "b.pdf"},
    {"chunk_id": "c", "text": "[Bulletin | Labour | PDF p.17]\nEmployment recorded strong growth in manufacturing.", "source": "b.pdf"},
]


@pytest.fixture(autouse=True)
def fake_corpus(monkeypatch):
    monkeypatch.setattr(rm, "macro_chunks", lambda: CHUNKS)
    rm._bm25.cache_clear()
    yield
    rm._bm25.cache_clear()


def test_bm25_returns_the_matching_chunk_first_and_drops_zero_scores():
    hits = rm.macro_bm25_search("how much did the franc lose against the dollar", 3)
    assert hits[0]["chunk_id"] == "b" and all(h["score"] > 0 for h in hits)
    assert rm.macro_bm25_search("zzz unmatched", 3) == []


def test_hybrid_rerank_fuses_both_rankers_then_reranks(monkeypatch):
    monkeypatch.setattr(rm, "macro_dense_search", lambda q, n: [dict(CHUNKS[2]), dict(CHUNKS[0])])
    seen = {}

    def fake_rerank(query, candidates, k):
        seen["ids"] = [c["chunk_id"] for c in candidates]
        return candidates[:k]

    monkeypatch.setattr(rm, "rerank", fake_rerank)
    out = rm.retrieve_macro("CPI inflation in August", k=1)
    assert len(out) == 1 and {"a", "c"} <= set(seen["ids"])  # candidates come from dense AND bm25


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        rm.retrieve_macro("anything", mode="magic")
