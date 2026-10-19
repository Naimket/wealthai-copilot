"""Retrieval over central-bank publications: dense + BM25 fused with RRF, then the cross-encoder reranker.

Same recipe as the KID retrieval (retrieval.py, whose BM25, RRF and reranker are reused) without fund detection:
macro questions are not about one product, and the macro chunks live in their own file and collection so they can
never leak into the fund rankings.
"""
import json
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient

from wealthai.ingestion.index import MACRO_COLLECTION, QDRANT_PATH, embedder
from wealthai.retrieval import BM25, CANDIDATES, rerank, rrf, tokenize

ROOT = Path(__file__).resolve().parents[2]
MACRO_CHUNKS_FILE = ROOT / "data" / "processed" / "macro_chunks.jsonl"


@lru_cache(maxsize=1)
def macro_chunks() -> list[dict]:
    return [json.loads(line) for line in MACRO_CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=1)
def _bm25() -> BM25:
    return BM25([tokenize(c["text"]) for c in macro_chunks()])


def macro_dense_search(query: str, n: int) -> list[dict]:
    vector = next(iter(embedder().query_embed(query))).tolist()
    client = QdrantClient(path=QDRANT_PATH)
    try:
        points = client.query_points(MACRO_COLLECTION, query=vector, limit=n).points
    finally:
        client.close()
    return [dict(p.payload, score=p.score) for p in points]


def macro_bm25_search(query: str, n: int) -> list[dict]:
    scores = _bm25().scores(tokenize(query))
    ranked = sorted(zip(macro_chunks(), scores), key=lambda x: -x[1])
    return [dict(c, score=s) for c, s in ranked if s > 0][:n]


MACRO_MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def retrieve_macro(query: str, k: int = 5, mode: str = "hybrid_rerank") -> list[dict]:
    if mode not in MACRO_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MACRO_MODES}")
    if mode == "dense":
        return macro_dense_search(query, k)
    if mode == "bm25":
        return macro_bm25_search(query, k)
    hybrid = rrf([macro_dense_search(query, CANDIDATES), macro_bm25_search(query, CANDIDATES)])
    if mode == "hybrid":
        return hybrid[:k]
    return rerank(query, hybrid[:CANDIDATES], k)
