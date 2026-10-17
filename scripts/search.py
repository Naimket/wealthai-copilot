"""Quick semantic search over the indexed KID chunks."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.ingestion.index import search  # noqa: E402

query = " ".join(sys.argv[1:]) or "What is the summary risk indicator?"
print(f"Query: {query}\n")
for hit in search(query, k=5):
    p = hit.payload
    print(f"{hit.score:.3f} | {p['product_name']} | {p['section']} | p.{p['page']}")
    print("    " + p["text"].replace("\n", " ")[:220])
