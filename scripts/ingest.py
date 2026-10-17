"""Parse -> chunk -> embed -> index every KID PDF found in data/raw."""
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.ingestion.chunk import chunk_doc  # noqa: E402
from wealthai.ingestion.index import build_index  # noqa: E402
from wealthai.ingestion.parse import parse_pdf  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "chunks.jsonl"


def main() -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {RAW_DIR}. Copy some KIDs there first.")

    all_chunks = []
    for pdf in pdfs:
        doc = parse_pdf(pdf)
        chunks = chunk_doc(doc)
        print(f"{pdf.name}: {len(doc.pages)} pages | ISIN {doc.isin} | {len(chunks)} chunks")
        print(f"   product : {doc.product_name}")
        print(f"   sections: {dict(Counter(c.section for c in chunks))}")
        all_chunks.extend(chunks)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {OUT_FILE.relative_to(ROOT)}")

    print("Embedding and indexing (first run downloads the model, please wait)...")
    t0 = time.perf_counter()
    n = build_index(all_chunks)
    print(f"Indexed {n} chunks in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
