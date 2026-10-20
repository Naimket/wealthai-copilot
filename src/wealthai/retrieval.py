"""Retrieval strategies for the ablation study.

Modes: dense | bm25 | hybrid | dense_rerank | hybrid_rerank | hybrid_filter_rerank | hybrid_perfund_rerank
  dense   : bge-small embeddings in Qdrant (the Phase 3 baseline)
  bm25    : lexical BM25 over the chunk text (exact words like 'Risk Indicator', '0.14%')
  hybrid  : dense + BM25 merged with Reciprocal Rank Fusion (RRF)
  *_rerank: top candidates re-scored by a cross-encoder (reads query and chunk together)
  *_filter: if the question names fund(s), only search those fund's documents
  *_perfund: like filter, but when several funds are named each one gets its own share of the
            results (so a comparison never loses one of the funds)
"""
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models

from wealthai.ingestion.index import COLLECTION, QDRANT_PATH, embedder

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20  # how many candidates each retriever contributes before fusion / reranking
MODES = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank", "hybrid_filter_rerank", "hybrid_perfund_rerank"]

TOKEN = re.compile(r"[a-z0-9&]+(?:\.[0-9]+)?")
STOPWORDS = {"the", "a", "an", "of", "is", "are", "and", "or", "in", "to", "for", "on", "per", "what", "which", "do", "does", "it", "its", "this", "that"}
GENERIC_NAME_TOKENS = {"ucits", "etf", "usd", "accu", "core", "fund", "the", "and"}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@lru_cache(maxsize=1)
def chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


@lru_cache(maxsize=1)
def bm25_index() -> BM25:
    return BM25([tokenize(c["text"]) for c in chunks()])


@lru_cache(maxsize=1)
def cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=RERANK_MODEL)


@lru_cache(maxsize=1)
def fund_keys() -> dict[str, set[str]]:
    """source file -> distinctive tokens of its product name ('iShares Core S&P 500 UCITS ETF' -> ishares, s&p, 500)."""
    names = {c["source"]: c["product_name"] for c in chunks()}
    return {src: set(tokenize(name)) - GENERIC_NAME_TOKENS for src, name in names.items()}


def detect_funds(query: str) -> list[str] | None:
    """Sources of the funds the question is about, or None if it names no fund (then search everything)."""
    q = set(tokenize(query))
    scores = {src: len(keys & q) / len(keys) for src, keys in fund_keys().items() if keys}
    best = max(scores.values(), default=0)
    if best < 0.5:
        return None
    return [src for src, s in scores.items() if s >= 0.5 and s >= best - 0.15]


# Questions that compare or rank the funds of the library without naming any of them ("which fund has the lowest...").
CROSS_FUND = re.compile(
    r"\b(?:which|what)\b.{0,40}\b(?:funds?|etfs?|kids?)\b|\b(?:highest|lowest|cheapest|most expensive|largest|smallest|best|worst|"
    r"compare|comparison|rank(?:ing)?|every fund|all (?:the |five )?funds|how many)\b", re.I)


def is_cross_fund(question: str) -> bool:
    """True if the question must be answered from ALL funds (it ranks or compares them) but names none of them."""
    return detect_funds(question) is None and bool(CROSS_FUND.search(question))


def dense_search(query: str, n: int, sources: list[str] | None = None) -> list[dict]:
    vector = next(iter(embedder().query_embed(query))).tolist()
    flt = None
    if sources:
        flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchAny(any=sources))])
    client = QdrantClient(path=QDRANT_PATH)
    try:
        points = client.query_points(COLLECTION, query=vector, limit=n, query_filter=flt).points
    finally:
        client.close()
    return [dict(p.payload, score=p.score) for p in points]


def bm25_search(query: str, n: int, sources: list[str] | None = None) -> list[dict]:
    scores = bm25_index().scores(tokenize(query))
    ranked = sorted(zip(chunks(), scores), key=lambda x: -x[1])
    return [dict(c, score=s) for c, s in ranked if s > 0 and (not sources or c["source"] in sources)][:n]


def rrf(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over rankers of 1/(k + rank). Needs no score normalisation."""
    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused[chunk["chunk_id"]] = fused.get(chunk["chunk_id"], 0.0) + 1.0 / (k + rank)
            by_id[chunk["chunk_id"]] = chunk
    return [dict(by_id[i], score=s) for i, s in sorted(fused.items(), key=lambda x: -x[1])]


def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    scores = list(cross_encoder().rerank(query, [c["text"] for c in candidates]))
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [dict(c, score=float(s)) for c, s in ranked[:k]]


def retrieve(query: str, k: int = 5, mode: str = "hybrid_filter_rerank", all_funds: bool = False) -> list[dict]:
    """all_funds=True (perfund modes only): treat every fund of the library as named, so each one gets its own share
    of the results. A top-k over the whole library can silently leave a fund out of a ranking."""
    sources = detect_funds(query) if ("filter" in mode or "perfund" in mode) else None
    if all_funds and "perfund" in mode:
        sources = list(fund_keys())
    if mode == "dense":
        return dense_search(query, k)
    if mode == "bm25":
        return bm25_search(query, k)
    if mode == "dense_rerank":
        return rerank(query, dense_search(query, CANDIDATES), k)
    hybrid = rrf([dense_search(query, CANDIDATES, sources), bm25_search(query, CANDIDATES, sources)])
    if mode == "hybrid":
        return hybrid[:k]
    if mode == "hybrid_perfund_rerank" and sources and len(sources) > 1:
        # ceil(k / n_funds), at least 2 chunks per fund, and one more per fund once k is widened by a retry (k > 5);
        # without that, the retry of a five-fund question (k=8) would return exactly the same 2 chunks per fund.
        per_fund = max(2, -(-k // len(sources)), 2 + max(0, k - 5) // 3)
        merged = []
        for src in sources:
            candidates = rrf([dense_search(query, CANDIDATES, [src]), bm25_search(query, CANDIDATES, [src])])
            merged += rerank(query, candidates[:CANDIDATES], per_fund)
        return sorted(merged, key=lambda c: -c["score"])
    if mode in ("hybrid_rerank", "hybrid_filter_rerank", "hybrid_perfund_rerank"):
        return rerank(query, hybrid[:CANDIDATES], k)
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
