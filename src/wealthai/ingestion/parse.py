"""PDF parsing for KIDs: text per page plus basic metadata (ISIN, product name)."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

SECTION_START = re.compile(
    r"(?i)^(what is this product|what are the risks|what happens if|what are the costs|"
    r"how long should i hold|how can i complain|other relevant information)"
)
ISIN_CANDIDATE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")


@dataclass
class ParsedPage:
    number: int
    text: str


@dataclass
class ParsedDoc:
    source: str
    product_name: str
    isin: str | None
    pages: list[ParsedPage]


def isin_is_valid(isin: str) -> bool:
    digits = "".join(str(int(ch, 36)) for ch in isin)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_isin(text: str) -> str | None:
    for candidate in ISIN_CANDIDATE.findall(text):
        if isin_is_valid(candidate):
            return candidate
    return None


def _clean_product_line(line: str) -> str:
    """'iShares X UCITS ETF (the "Fund"), USD Accu (the "Share Class"), ISIN: ...' -> 'iShares X UCITS ETF USD Accu'."""
    line = re.sub(r"(?i)^product\s*:\s*", "", line)  # some issuers repeat the label
    fund = re.split(r"\s*\((?:the|\u201c|\")", line, maxsplit=1)[0].strip(" ,")
    share_class = re.search(r"\),\s*(.+?)\s*\(the\s*[\u201c\"]Share Class", line)
    return f"{fund} {share_class.group(1)}" if share_class else fund


def guess_product_name(first_page: str, fallback: str) -> str:
    """KIDs state the product name right after the 'Product' label."""
    lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if line.lower() in {"product", "produit", "produkt"} and i + 1 < len(lines):
            return _clean_product_line(lines[i + 1])
        match = re.match(r"(?i)^product\s*:?\s+(.+)$", line)
        if match:
            return _clean_product_line(match.group(1))
    return fallback


def page_text(page) -> str:
    """Page text in reading order that respects KID section headings.

    PDF stream order is not visual order: in two-column layouts (e.g. Vanguard) a
    heading can come AFTER the text that sits below it. We keep the stream order but
    stably re-sort blocks by 'how many section headings are above me on the page'.
    """
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    heading_ys = [b[1] for b in blocks if SECTION_START.match(b[4].strip())]

    def rank(block):
        below = sum(1 for y in heading_ys if y <= block[1] + 2)
        is_heading = SECTION_START.match(block[4].strip()) is not None
        return (below, 0 if is_heading else 1)  # heading first, then its content in stream order

    return "".join(b[4] for b in sorted(blocks, key=rank))  # sorted() is stable


def parse_pdf(path: Path) -> ParsedDoc:
    pages = []
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            # NFKC turns non-breaking spaces and ligatures into plain characters.
            text = unicodedata.normalize("NFKC", page_text(page))
            pages.append(ParsedPage(number=number, text=text))
    full_text = "\n".join(p.text for p in pages)
    return ParsedDoc(
        source=path.name,
        product_name=guess_product_name(pages[0].text, fallback=path.stem),
        isin=find_isin(full_text),
        pages=pages,
    )
