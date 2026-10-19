"""Parse -> chunk -> embed -> index the central-bank PDFs in data/raw/macro (separate from the KID index).

Runs locally and uses no API quota. The KID index (scripts/ingest.py) is not touched.
"""
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.ingestion.index import MACRO_COLLECTION, build_index  # noqa: E402
from wealthai.ingestion.macro import chunk_macro, parse_macro_pdf  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "macro"
OUT_FILE = ROOT / "data" / "processed" / "macro_chunks.jsonl"


def main() -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {RAW_DIR}.")
    all_chunks = []
    for pdf in pdfs:
        doc = parse_macro_pdf(pdf)
        chunks = chunk_macro(doc)
        headings = sum(b.heading_level is not None for b in doc.blocks)
        print(f"{pdf.name}: {len(doc.blocks)} text blocks | {headings} headings | {len(chunks)} chunks")
        for section, n in Counter(c.section for c in chunks).items():
            print(f"   {n:2d}  {section}")
        all_chunks.extend(chunks)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {OUT_FILE.relative_to(ROOT)}")

    t0 = time.perf_counter()
    n = build_index(all_chunks, collection=MACRO_COLLECTION)
    print(f"Indexed {n} chunks in collection '{MACRO_COLLECTION}' in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
