"""Retrieval ablation on the macro set (no LLM calls, no API quota): hit@k and MRR per retrieval mode.

A question counts as a hit when one of the top-k chunks matches the question's answer pattern.
Usage: uv run python scripts/eval_retrieval_macro.py [--k 5]
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.retrieval_macro import MACRO_MODES, retrieve_macro  # noqa: E402

K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 5
SET = ROOT / "data" / "golden" / "macro_set.jsonl"
# Comparatives are answered by combining two facts: their pattern describes the answer, not one chunk.
items = [json.loads(line) for line in SET.read_text(encoding="utf-8").splitlines() if line.strip()]
items = [i for i in items if i["answerable"] and i["type"] != "comparative"]


def norm(text):
    return re.sub(r"\s+", " ", text.replace(" ", " "))


def first_hit(item, hits):
    for rank, hit in enumerate(hits, start=1):
        if any(re.search(p, norm(hit["text"]), re.I) for p in item["match"]):
            return rank
    return None


print(f"Macro retrieval, {len(items)} answerable single-fact questions, k={K}\n")
print(f"{'mode':16s} {'hit@' + str(K):>7s} {'MRR':>7s} {'s/query':>8s}")
misses = {}
for mode in MACRO_MODES:
    t0 = time.perf_counter()
    ranks = [first_hit(i, retrieve_macro(i["question"], k=K, mode=mode)) for i in items]
    elapsed = (time.perf_counter() - t0) / len(items)
    hits = [r for r in ranks if r]
    mrr = sum(1 / r for r in hits) / len(items)
    print(f"{mode:16s} {100 * len(hits) / len(items):6.1f}% {mrr:7.3f} {elapsed:8.2f}")
    misses[mode] = [i["id"] for i, r in zip(items, ranks) if r is None]
print()
for mode, ids in misses.items():
    print(f"{mode}: missed {ids}")
