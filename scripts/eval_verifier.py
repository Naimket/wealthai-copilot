"""Stress test of the `verify` node: does it catch answers that are wrong, and does it leave correct ones alone?

In the normal pipeline the verifier almost never fires because the generator is usually faithful, so its
real detection rate is unknown. This script feeds the real node (wealthai.agent.graph.verify) with a small
labelled set: faithful answers and answers corrupted on purpose (changed number, swapped fund, invented fee,
wrong ranking, reversed fact, unsupported addition, wrong arithmetic).

The passages in data/golden/verifier_set.jsonl are short excerpts written from the real KIDs (same figures,
trimmed text), so every label can be checked by eye.

Metrics (positive = "answer is NOT supported"):
  detection rate  = corrupted answers flagged / corrupted answers      (recall)
  false alarms    = faithful answers flagged / faithful answers        (1 - specificity)

Usage: uv run python scripts/eval_verifier.py [--run N] [--redo v01,v02]
Cost: one LLM call per case (16). Results are cached per model and run in data/eval/verifier_<model>_r<N>.jsonl.
Small set: report counts, not percentages, and treat them as a smoke test rather than a benchmark.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai import llm  # noqa: E402
from wealthai.agent.graph import verify  # noqa: E402


def _option(name: str, default: str = "") -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


RUN = _option("--run", "1")
GOLDEN = ROOT / "data" / "golden" / "verifier_set.jsonl"
CACHE = ROOT / "data" / "eval" / f"verifier_{re.sub(r'[^A-Za-z0-9.-]+', '_', llm.MODEL)}_r{RUN}.jsonl"


def summarise(cases: list[dict], rows: dict[str, dict]) -> dict:
    """Pure function (tested without a model): counts per corruption kind plus overall detection / false alarms."""
    corrupted = [c for c in cases if c["id"] in rows and not c["supported"]]
    faithful = [c for c in cases if c["id"] in rows and c["supported"]]
    caught = [c for c in corrupted if not rows[c["id"]]["supported"]]
    alarms = [c for c in faithful if not rows[c["id"]]["supported"]]
    by_kind = defaultdict(lambda: [0, 0])
    for c in corrupted:
        by_kind[c["kind"]][1] += 1
        by_kind[c["kind"]][0] += not rows[c["id"]]["supported"]
    return {"corrupted": len(corrupted), "caught": len(caught), "faithful": len(faithful), "false_alarms": len(alarms),
            "missed": [c["id"] for c in corrupted if c not in caught], "alarm_ids": [c["id"] for c in alarms],
            "by_kind": dict(by_kind)}


def main() -> None:
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["id"]] = row
    for qid in _option("--redo").split(","):
        cache.pop(qid.strip(), None)
    print(f"MODEL: {llm.MODEL} | run {RUN} | cases: {len(cases)} | cached: {len(cache)}\n")
    for c in cases:
        if c["id"] in cache:
            continue
        state = {"question": c["question"], "answer": c["answer"], "chunks": [{"text": t} for t in c["passages"]]}
        try:
            out = verify(state)
        except Exception as exc:  # e.g. daily quota: keep what we have, re-run to continue
            print(f"Stopping at {c['id']}: {type(exc).__name__}: {str(exc)[:150]}\nShowing partial results; re-run to continue.")
            break
        row = {"id": c["id"], "model": llm.MODEL, "supported": out["supported"], "unsupported": out["unsupported"]}
        cache[c["id"]] = row
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{c['id']} [{c['kind']}] supported={out['supported']}")

    s = summarise(cases, cache)
    print(f"\n=== VERIFIER STRESS TEST | model: {llm.MODEL} | run {RUN} | {len(cache)}/{len(cases)} cases ===")
    print(f"detection of corrupted answers: {s['caught']}/{s['corrupted']}")
    for kind, (hit, n) in s["by_kind"].items():
        print(f"  {kind:22s} {hit}/{n}")
    print(f"false alarms on faithful answers: {s['false_alarms']}/{s['faithful']}")
    if s["missed"]:
        print("\nMISSED (corrupted but judged supported):")
        for cid in s["missed"]:
            c = next(c for c in cases if c["id"] == cid)
            print(f"  {cid} [{c['kind']}] {c['answer'][:140]}")
    if s["alarm_ids"]:
        print("\nFALSE ALARMS (faithful but flagged):")
        for cid in s["alarm_ids"]:
            print(f"  {cid}: {cache[cid]['unsupported'][:200]}")


if __name__ == "__main__":
    main()
