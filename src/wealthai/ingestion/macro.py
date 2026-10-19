"""Parsing and chunking of central-bank publications (SNB press releases and bulletins).

KIDs have a fixed section structure; macro reports do not. Here the structure comes from the PDF's own bookmarks
(table of contents) when it has them, and the reading order has to be rebuilt: the bulletin is laid out in two columns
with charts next to the text, and chart axes, legends and page headers must not end up in the index.
"""
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from wealthai.ingestion.chunk import Chunk

MAX_CHARS = 1200
MIN_PROSE_WORDS = 10  # blocks with fewer words are chart labels, axis ticks, legends or page headers
MIN_WORD_SHARE = 0.6  # share of tokens that are plain words (no digits): drops chart data series like '3.2 2.1 1.6 Q1 Q2'
_WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
COLUMN_SPLIT = 0.45  # x0 beyond this fraction of the page width = right-hand column

# Explicit metadata: the PDFs do not state their title in a machine-readable way.
MACRO_DOCS = {
    "snb_monetary_policy_assessment_2026_09_24.pdf": "SNB Monetary policy assessment of 24 September 2026",
    "snb_quarterly_bulletin_2026_3.pdf": "SNB Quarterly Bulletin 3/2026 (September 2026)",
}
PAGE_FOOTER = re.compile(r"(?i)^\d+\s+quarterly bulletin|^quarterly bulletin \d|^source\(s\)|^page \d+/\d+|^chart \d")


def clean_text(text: str) -> str:
    """Normalise a block of PDF text.

    The bulletin's font leaves a stray space after some fi/fl/ff glyph pairs ('infl ation', 'signifi cantly',
    'eff ect'). Left in, they break both keyword and embedding search for the most frequent word of the corpus.
    """
    text = unicodedata.normalize("NFKC", text).replace(" ", " ").replace("­", "")
    text = re.sub(r"(fi|fl)\s+(?=[a-z])", r"\1", text)  # no English word ends in 'fi' or 'fl' before a lowercase word
    text = re.sub(r"\b(e|a|di|E|A|Di)ff\s+(?=[a-z])", r"\1ff", text)  # eff ect, aff ected, diff erence ('staff is' stays)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def is_prose(text: str) -> bool:
    tokens = [w for w in text.split() if w != "\u2192"]  # '→' marks glossary cross-references
    words = sum(bool(_WORD.fullmatch(w.strip(".,;:()%'\"“”‘’"))) for w in tokens)
    return len(tokens) >= MIN_PROSE_WORDS and words / len(tokens) >= MIN_WORD_SHARE


def _norm(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^\d+\s+", "", title.strip())).casefold()


@dataclass
class MacroBlock:
    page: int
    text: str
    heading_level: int | None = None  # set when the block is a TOC heading


@dataclass
class MacroDoc:
    source: str
    title: str
    blocks: list[MacroBlock]


def _text_blocks(page) -> list[dict]:
    """Text blocks of a page in reading order (left column top to bottom, then right column), with their font."""
    width = page.rect.width
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines = ["".join(s["text"] for s in line["spans"]) for line in block["lines"]]
        spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        out.append({"x0": block["bbox"][0], "y0": block["bbox"][1], "text": "\n".join(lines),
                    "font": spans[0]["font"], "size": spans[0]["size"]})
    return sorted(out, key=lambda b: (b["x0"] > width * COLUMN_SPLIT, round(b["y0"]), b["x0"]))


# (font name fragment, minimum size). SNB section headings are 'Pro' Medium capitals (9.5+ pt) or Thin chapter titles
# (18+ pt). Chart titles use another family (W04) and run-in leads are Bold, so neither is mistaken for a heading.
# Another publisher needs its own style list; None accepts any font and relies on the bookmarks alone.
SNB_HEADING_STYLES = [("UniversNextPro-Medium", 9.4), ("UniversNextPro-Thin", 18.0)]


def _is_heading_font(block: dict, styles) -> bool:
    if styles is None:
        return True
    return any(fragment in block["font"] and block["size"] >= min_size for fragment, min_size in styles)


def parse_macro_pdf(path: Path, heading_styles=SNB_HEADING_STYLES) -> MacroDoc:
    title = MACRO_DOCS.get(path.name, path.stem)
    with pymupdf.open(path) as pdf:
        toc = pdf.get_toc()
        # Skip cover and contents: everything before the first real chapter.
        chapters = [p for level, name, p in toc if level == 1 and _norm(name) != "contents"]
        first_page = min(chapters) if chapters else 1
        # The bookmarks point to the first page of a printed spread, so a heading can sit one PDF page later.
        pending = [(page, _norm(name), level) for level, name, page in toc if _norm(name) != "contents"]

        blocks: list[MacroBlock] = []
        for number, page in enumerate(pdf, start=1):
            if number < first_page:
                continue
            for b in _text_blocks(page):
                lines = b["text"].split("\n")
                if _is_heading_font(b, heading_styles):
                    # The heading may share a block with the paragraph that follows, and may wrap onto 2-3 lines.
                    for k in range(1, min(3, len(lines)) + 1):
                        title_text = clean_text(" ".join(lines[:k]))
                        match = next((i for i, (tp, name, _) in enumerate(pending)
                                      if tp <= number <= tp + 1 and name == _norm(title_text)), None)
                        if match is not None:
                            level = pending.pop(match)[2]  # each bookmark is used once ('Inflation expectations' twice)
                            blocks.append(MacroBlock(number, title_text, heading_level=level))
                            lines = lines[k:]
                            break
                text = clean_text("\n".join(lines))
                if is_prose(text) and not PAGE_FOOTER.match(text):
                    blocks.append(MacroBlock(number, text))
    return MacroDoc(source=path.name, title=title, blocks=blocks)


def _split_long(text: str) -> list[str]:
    """Split a block that exceeds MAX_CHARS at sentence ends."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > MAX_CHARS:
            parts.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_macro(doc: MacroDoc) -> list[Chunk]:
    """Paragraph blocks are packed up to MAX_CHARS; a heading always starts a new chunk (never crossed)."""
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (level, title) of the open headings
    buffer: list[str] = []
    buffer_page = 1

    def flush():
        nonlocal buffer
        if not buffer:
            return
        section = " > ".join(t for _, t in stack) or doc.title
        label = doc.title if section == doc.title else f"{doc.title} | {section}"  # no 'title | title' for flat documents
        text = f"[{label} | PDF p.{buffer_page}]\n" + " ".join(buffer)
        chunk_id = hashlib.sha1(f"{doc.source}:{len(chunks)}".encode()).hexdigest()[:12]
        chunks.append(Chunk(chunk_id, text, doc.source, doc.title, None, section, buffer_page))
        buffer = []

    for block in doc.blocks:
        if block.heading_level is not None:
            flush()
            while stack and stack[-1][0] >= block.heading_level:
                stack.pop()
            stack.append((block.heading_level, block.text))
            continue
        for part in _split_long(block.text):
            if buffer and sum(len(s) + 1 for s in buffer) + len(part) > MAX_CHARS:
                flush()
            if not buffer:
                buffer_page = block.page
            buffer.append(part)
    flush()
    return chunks
