"""End-to-end evaluation of the macro route on data/golden/macro_set.jsonl (deterministic checks, no LLM judge).

Each question goes through the whole multi-source agent (run_multi), so the router is evaluated as well:
an answerable question counts as correct only if it was routed to `macro`, the agent answered, and the answer
matches the expected fact. An unanswerable question is correct if the system declined or abstained.

Usage: uv run python scripts/eval_macro.py [--run N] [--redo m01,m02]
Answers are cached per model and run in data/eval/macro_<model>_r<N>.jsonl, so a run stopped by a quota error can be
repeated to continue. Always read the printed MODEL line before quoting numbers.
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

from wealthai import llm  # noqa: E402
from wealthai.agent.multi import run_multi  # noqa: E402

GOLDEN = ROOT / "data" / "golden" / "macro_set.jsonl"


def _option(name: str, default: str = "") -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


RUN = _option("--run", "1")
CACHE = ROOT / "data" / "eval" / f"macro_{re.sub(r'[^A-Za-z0-9.-]+', '_', llm.MODEL)}_r{RUN}.jsonl"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" ", " "))


def any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, norm(text), re.I) for p in patterns)


def verdict(item: dict, row: dict) -> tuple[bool, bool, str]:
    """(route_ok, correct, reason). Unanswerable items only need a decline/abstention."""
    if not item["answerable"]:
        ok = not row["answerable"]
        return True, ok, "" if ok else "answered a question the documents cannot answer"
    route_ok = row["route"] == "macro"
    if not route_ok:
        return False, False, f"routed to {row['route']!r}"
    if not row["answerable"]:
        return True, False, "abstained"
    ok = any_match(item["match"], row["answer"])
    return True, ok, "" if ok else "answer does not contain the expected fact"


items = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
CACHE.parent.mkdir(parents=True, exist_ok=True)
cache = {}
if CACHE.exists():
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["id"]] = row
for qid in _option("--redo").split(","):
    cache.pop(qid.strip(), None)

print(f"MODEL: {llm.MODEL} | run {RUN} | questions: {len(items)} | cached: {len(cache)}\n")
for item in items:
    if item["id"] in cache:
        continue
    t0 = time.perf_counter()
    try:
        r = run_multi(item["question"])
    except Exception as exc:  # e.g. daily quota exhausted: keep what we have, re-run later to continue
        print(f"Stopping at {item['id']}: {type(exc).__name__}: {str(exc)[:150]}\nShowing partial results; re-run to continue.")
        break
    row = {"id": item["id"], "model": llm.MODEL, "seconds": round(time.perf_counter() - t0, 1),
           **{k: r[k] for k in ("question", "answer", "answerable", "route", "sources", "contexts", "trace")}}
    cache[item["id"]] = row
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{item['id']} [{item['type']}] route={r['route']} in {row['seconds']}s")

done = [i for i in items if i["id"] in cache]
stats = defaultdict(lambda: {"n": 0, "route": 0, "correct": 0})
failures = []
for item in done:
    route_ok, correct, reason = verdict(item, cache[item["id"]])
    s = stats[item["type"]]
    s["n"] += 1
    s["route"] += route_ok
    s["correct"] += correct
    if not correct:
        failures.append((item, cache[item["id"]], reason))

print(f"\n=== MACRO END-TO-END | model: {llm.MODEL} | run {RUN} | {len(done)}/{len(items)} questions ===")
print(f"{'type':14s} {'n':>3s} {'routed':>8s} {'correct':>9s}")
for t, s in stats.items():
    print(f"{t:14s} {s['n']:3d} {s['route']:5d}/{s['n']:<2d} {s['correct']:6d}/{s['n']:<2d} = {100 * s['correct'] / s['n']:5.1f}%")
total = sum(s["n"] for s in stats.values())
print(f"{'TOTAL':14s} {total:3d} {sum(s['route'] for s in stats.values()):5d}/{total:<2d} "
      f"{sum(s['correct'] for s in stats.values()):6d}/{total:<2d}")
if done:
    print(f"mean latency: {sum(cache[i['id']]['seconds'] for i in done) / len(done):.1f}s per question")

print("\n=== FAILURES ===")
for item, row, reason in failures:
    print(f"\n{item['id']} [{item['type']}] {item['question']}")
    print(f"  reason  : {reason}")
    print(f"  expected: {item['reference']}")
    print(f"  got     : {row['answer'][:300]}")
