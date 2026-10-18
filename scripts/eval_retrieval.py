"""Retrieval-only ablation on the golden set. Makes NO LLM calls (free, ~1-2 min).

For every answerable question with a single expected PDF we check whether the top-k chunks contain
the right document AND the fact (regex match). Reports hit@k and MRR per retrieval mode.
Usage: uv run python scripts/eval_retrieval.py [k]
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.retrieval import MODES, retrieve  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
golden = [json.loads(l) for l in (ROOT / "data" / "golden" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
items = [g for g in golden if g["answerable"] and g["source"]]
print(f"{len(items)} single-document questions, k={K}\n")


def fact_in(item, text):
    t = re.sub(r"\s+", " ", text.replace("\u00a0", " "))
    return any(re.search(p, t, re.I) for p in item["match"])


results = {}
misses = {}
for mode in MODES:
    t0 = time.perf_counter()
    per_type = defaultdict(lambda: {"hit": [], "rr": [], "doc": []})
    miss = []
    for item in items:
        chunks = retrieve(item["question"], k=K, mode=mode)
        rank = next((i for i, c in enumerate(chunks, 1) if c["source"] == item["source"] and fact_in(item, c["text"])), None)
        for key in (item["type"], "ALL"):
            per_type[key]["hit"].append(rank is not None)
            per_type[key]["rr"].append(1 / rank if rank else 0.0)
            per_type[key]["doc"].append(any(c["source"] == item["source"] for c in chunks))
        if rank is None:
            miss.append(item)
    results[mode] = per_type
    misses[mode] = miss
    print(f"  {mode:22s} done in {time.perf_counter() - t0:5.1f}s")

print(f"\n=== RETRIEVAL ABLATION (hit@{K} = right PDF and the fact is in a retrieved chunk) ===")
print(f"{'mode':22s} {'type':9s} {'n':>3s} {'hit@k':>7s} {'MRR':>6s} {'right-PDF@k':>12s}")
for mode in MODES:
    for t in ("factual", "hard", "ALL"):
        r = results[mode].get(t)
        if r:
            n = len(r["hit"])
            print(f"{mode:22s} {t:9s} {n:3d} {100 * sum(r['hit']) / n:6.1f}% {sum(r['rr']) / n:6.3f} {100 * sum(r['doc']) / n:11.1f}%")
    print()

best = max(MODES, key=lambda m: sum(results[m]["ALL"]["hit"]))
print(f"=== STILL MISSED by the best mode ({best}) ===")
for item in misses[best]:
    print(f"{item['id']} {item['question']}")
