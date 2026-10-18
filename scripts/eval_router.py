"""Evaluate the multi-source router on data/golden/router_set.jsonl with deterministic checks (no LLM judge).

Usage: uv run python scripts/eval_router.py [--fresh] [--redo r07,r21]

Answers are cached per MODEL in data/eval/router_<model>.jsonl, so a run that stops on a quota error can simply
be repeated, and runs with different models never mix. Always read the printed MODEL line before quoting numbers.
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
from wealthai.evaluation.router_checks import check_item  # noqa: E402
from wealthai.tools.sql import run_query  # noqa: E402

GOLDEN = ROOT / "data" / "golden" / "router_set.jsonl"
CACHE = ROOT / "data" / "eval" / f"router_{re.sub(r'[^A-Za-z0-9.-]+', '_', llm.MODEL)}.jsonl"

items = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
CACHE.parent.mkdir(parents=True, exist_ok=True)
if "--fresh" in sys.argv and CACHE.exists():
    CACHE.unlink()
cache = {}
if CACHE.exists():
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["id"]] = row
if "--redo" in sys.argv:
    for qid in sys.argv[sys.argv.index("--redo") + 1].split(","):
        cache.pop(qid.strip(), None)


def db_counts() -> tuple:
    return (run_query("SELECT COUNT(*) FROM clients").rows[0][0], run_query("SELECT COUNT(*) FROM holdings").rows[0][0])


counts_before = db_counts()
print(f"MODEL: {llm.MODEL} | questions: {len(items)} | cached: {len(cache)}\n")

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
           **{k: r[k] for k in ("question", "answer", "answerable", "route", "sql", "sources", "trace")}}
    cache[item["id"]] = row
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{item['id']} [{item['category']}] route={r['route']} in {row['seconds']}s")

done = [i for i in items if i["id"] in cache]
by_cat = defaultdict(list)
failures = []
for item in done:
    verdict = check_item(item, cache[item["id"]])
    by_cat[item["category"]].append(verdict)
    if not verdict["passed"]:
        failures.append((item, cache[item["id"]], verdict))

print(f"\n=== ROUTER EVALUATION | model: {llm.MODEL} | {len(done)}/{len(items)} questions ===")
print(f"{'category':15s} {'n':>3s} {'route ok':>9s} {'answer ok':>10s} {'passed':>7s}")
for cat, vs in by_cat.items():
    n = len(vs)
    print(f"{cat:15s} {n:3d} {sum(v['route_ok'] for v in vs):6d}/{n:<2d} {sum(v['answer_ok'] for v in vs):7d}/{n:<2d} {sum(v['passed'] for v in vs):4d}/{n:<2d}")
allv = [v for vs in by_cat.values() for v in vs]
print(f"{'TOTAL':15s} {len(allv):3d} {sum(v['route_ok'] for v in allv):6d}/{len(allv):<2d} "
      f"{sum(v['answer_ok'] for v in allv):7d}/{len(allv):<2d} {sum(v['passed'] for v in allv):4d}/{len(allv):<2d}")
if done:
    print(f"mean latency: {sum(cache[i['id']]['seconds'] for i in done) / len(done):.1f}s per question")

counts_after = db_counts()
print(f"\nDatabase integrity (clients, holdings): before {counts_before} -> after {counts_after} "
      f"{'OK' if counts_before == counts_after else 'CHANGED!'}")

print("\n=== FAILURES ===")
for item, row, verdict in failures:
    print(f"\n{item['id']} [{item['category']}] {item['question']}")
    print(f"  route   : {row['route']!r} (accepted: {item['routes']})")
    print(f"  reasons : {'; '.join(verdict['reasons'])}")
    print(f"  answer  : {row['answer'][:300]}")
    if row["sql"]:
        print(f"  sql     : {row['sql'][:300]}")
