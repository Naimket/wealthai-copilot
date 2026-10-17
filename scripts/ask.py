"""Ask the naive RAG a question: uv run python scripts/ask.py "your question"."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.rag.naive import answer_question  # noqa: E402

question = " ".join(sys.argv[1:]) or "What are the ongoing costs of the Vanguard FTSE All-World UCITS ETF?"
r = answer_question(question)
print(f"Q: {question}\n")
print(f"A: {r['answer']}\n")
print(f"answerable={r['answerable']} | citations={r['citations']}")
for i, (src, sec) in enumerate(zip(r["sources"], r["sections"]), start=1):
    print(f"  [{i}] {src} | {sec}")
