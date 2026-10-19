"""Ragas evaluation of the cached RAG answers (faithfulness + answer relevancy).

Uses Gemini (flash-lite) as the judge and local fastembed embeddings, so it stays free.
Reads  data/eval/naive_answers.jsonl  (made by eval_naive.py) and writes data/eval/ragas_naive.csv.
Usage: uv run python scripts/eval_ragas.py [answers_file] [output_csv] [--set golden|blind]
"""
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import wealthai.llm as wllm  # noqa: E402,F401  (loads .env into os.environ)
from fastembed import TextEmbedding  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import AnswerRelevancy, Faithfulness  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "golden"
_args = [a for i, a in enumerate(sys.argv[1:], 1) if a != "--set" and sys.argv[i - 1] != "--set"]
ANSWERS = Path(_args[0]) if len(_args) > 0 else ROOT / "data" / "eval" / "naive_answers.jsonl"
OUT = Path(_args[1]) if len(_args) > 1 else ROOT / "data" / "eval" / "ragas_naive.csv"
GOLDEN = ROOT / "data" / "golden" / f"{SET}_set.jsonl"


class LocalEmbeddings(Embeddings):
    """LangChain-style embeddings backed by fastembed (CPU, no API calls)."""

    def __init__(self):
        self._engine = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts):
        return [v.tolist() for v in self._engine.embed(texts)]

    def embed_query(self, text):
        return next(iter(self._engine.embed([text]))).tolist()


golden = {}
for line in GOLDEN.read_text(encoding="utf-8").splitlines():
    if line.strip():
        item = json.loads(line)
        golden[item["id"]] = item
rows = [json.loads(l) for l in ANSWERS.read_text(encoding="utf-8").splitlines() if l.strip()]
# Abstentions ("the documents do not contain...") have no claims to verify: Ragas only scores answered questions.
rows = [r for r in rows if r["answerable"] and r["id"] in golden]
print(f"Scoring {len(rows)} answered questions with Ragas (set={SET}, judge={os.environ.get('GEMINI_MODEL', 'gemini-flash-lite-latest')})...")

judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest"),
    google_api_key=os.environ["GEMINI_API_KEY"], temperature=0))
dataset = EvaluationDataset.from_list([
    {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"]} for r in rows])

result = evaluate(
    dataset,
    metrics=[Faithfulness(), AnswerRelevancy(strictness=1)],  # Gemini returns 1 candidate per call
    llm=judge,
    embeddings=LangchainEmbeddingsWrapper(LocalEmbeddings()),
    run_config=RunConfig(max_workers=2, timeout=120, max_retries=6),
)
df = result.to_pandas()
df.insert(0, "id", [r["id"] for r in rows])
df.insert(1, "type", [golden[r["id"]]["type"] for r in rows])
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("\n=== RAGAS (mean per question type; NaN = metric could not be computed) ===")
print(df.groupby("type")[["faithfulness", "answer_relevancy"]].mean().round(3).to_string())
print("\noverall:", df[["faithfulness", "answer_relevancy"]].mean().round(3).to_dict())
print(f"NaN count: {int(df[['faithfulness', 'answer_relevancy']].isna().sum().sum())} | saved {OUT}")

