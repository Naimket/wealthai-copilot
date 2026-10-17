"""Print a one-line overview of every chunk whose product name contains the given text."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
needle = (sys.argv[1] if len(sys.argv) > 1 else "").lower()

with (ROOT / "data" / "processed" / "chunks.jsonl").open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        c = json.loads(line)
        if needle in c["product_name"].lower():
            body = c["text"].split("\n", 1)[1].replace("\n", " ")
            print(f"{i:3d} {c['section']:15s} p{c['page']} {len(c['text']):5d}ch | {body[:100]}")
