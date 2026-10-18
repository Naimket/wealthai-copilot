"""Ask the multi-source agent (KIDs + client database): uv run python scripts/ask_multi.py "your question"."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.agent.multi import run_multi  # noqa: E402

question = " ".join(sys.argv[1:]) or "Which funds does Alice Brunner hold and what are their ongoing costs?"
r = run_multi(question)
print(f"Q: {question}\n")
print("TRACE:")
for step in r["trace"]:
    print(f"  - {step}")
print(f"\nA: {r['answer']}\n")
print(f"route={r['route']} | answerable={r['answerable']} | sources={sorted(set(r['sources']))}")
