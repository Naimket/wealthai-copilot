"""Run the naive RAG on the golden set and report cheap deterministic metrics.

Answers are cached in data/eval/naive_answers.jsonl, so re-running only calls Gemini
for questions that are still missing (the free tier is slow and rate-limited).
Usage: uv run python scripts/eval_naive.py [--fresh]
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

from wealthai.rag.naive import answer_question as _naive_answer  # noqa: E402


def answer_question(question, mode):
    """mode 'agent' = the LangGraph agent; any other mode = single-pass RAG with that retrieval strategy."""
    if mode == "agent":
        from wealthai.agent.graph import run_agent
        return run_agent(question)
    return _naive_answer(question, mode=mode)


GOLDEN = ROOT / "data" / "golden" / "golden_set.jsonl"
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "dense"
CACHE = ROOT / "data" / "eval" / ("naive_answers.jsonl" if MODE == "dense" else f"answers_{MODE}.jsonl")
PAUSE_SECONDS = 0  # wealthai.llm already spaces calls to respect the free-tier requests-per-minute limit

golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
CACHE.parent.mkdir(parents=True, exist_ok=True)
if "--fresh" in sys.argv and CACHE.exists():
    CACHE.unlink()
cache = {}
if CACHE.exists():
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["id"]] = row

if "--redo" in sys.argv:  # e.g. --redo q51,q42 : ask these questions again (the newest line in the cache wins)
    for qid in sys.argv[sys.argv.index("--redo") + 1].split(","):
        cache.pop(qid.strip(), None)

for item in golden:
    if item["id"] in cache:
        continue
    t0 = time.perf_counter()
    try:
        result = answer_question(item["question"], MODE)
    except Exception as exc:  # e.g. daily quota exhausted: keep what we have and report partial results
        print(f"Stopping at {item['id']}: {type(exc).__name__}: {str(exc)[:150]}\nShowing partial results; re-run later to continue.")
        break
    result["id"] = item["id"]
    cache[item["id"]] = result
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"{item['id']} done in {time.perf_counter() - t0:.1f}s")
    time.sleep(PAUSE_SECONDS)

golden = [g for g in golden if g["id"] in cache]


def norm(text):
    return re.sub(r"\s+", " ", text.replace("\u00a0", " "))


def any_match(patterns, text):
    return any(re.search(p, norm(text), re.I) for p in patterns)


stats = defaultdict(lambda: defaultdict(list))
failures = []
for item in golden:
    r = cache[item["id"]]
    t = item["type"]
    if item["answerable"]:
        if item["source"]:  # retrieval: right document AND the fact is inside a retrieved chunk
            hit = any(s == item["source"] and any_match(item["match"], c)
                      for s, c in zip(r["sources"], r["contexts"]))
            stats[t]["retrieval_hit"].append(hit)
        correct = r["answerable"] and any_match(item["match"], r["answer"])
        stats[t]["answer_correct"].append(correct)
        if not correct:
            failures.append((item, r))
    else:
        correct = not r["answerable"]  # should abstain
        stats[t]["correct_abstention"].append(correct)
        if not correct:
            failures.append((item, r))

print("\n=== NAIVE RAG BASELINE ===")
for t, metrics in stats.items():
    for name, values in metrics.items():
        print(f"{t:13s} {name:18s} {sum(values):2d}/{len(values):2d} = {100 * sum(values) / len(values):5.1f}%")

print("\n=== FAILURES ===")
for item, r in failures:
    print(f"\n{item['id']} [{item['type']}] {item['question']}")
    print(f"  expected: {item['reference']}")
    print(f"  got     : {r['answer'][:300]}")
    print(f"  sources : {r['sources']}")

if MODE == "agent":
    rows = list(cache.values())
    n = len(rows)
    print("\n=== AGENT BEHAVIOUR ===")
    print(f"questions: {n} | refused as advice: {sum(any('refuse' in t for t in r['trace']) for r in rows)}"
          f" | retried retrieval: {sum(sum('rewrite' in t for t in r['trace']) > 1 for r in rows)}"
          f" | regenerated after verify: {sum(any('draft 2' in t for t in r['trace']) for r in rows)}"
          f" | abstained: {sum(any(t == 'abstain' for t in r['trace']) for r in rows)}")
