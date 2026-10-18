"""Ask the LangGraph agent and print its decision trace: uv run python scripts/ask_agent.py "your question"."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.agent.graph import run_agent  # noqa: E402

question = " ".join(sys.argv[1:]) or "What is the TER of the Vanguard FTSE All-World UCITS ETF?"
r = run_agent(question)
print(f"Q: {question}\n")
print("TRACE:")
for step in r["trace"]:
    print(f"  - {step}")
print(f"\nA: {r['answer']}\n")
print(f"answerable={r['answerable']} | citations={r['citations']} | sources={sorted(set(r['sources']))}")
