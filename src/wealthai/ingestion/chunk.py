"""Section-aware chunking for PRIIPs KIDs."""
import hashlib
import re
from dataclasses import dataclass

from wealthai.ingestion.parse import ParsedDoc

SECTION_HEADINGS = [
    ("product", re.compile(r"^what is this product", re.I)),
    ("risk", re.compile(r"^what are the risks", re.I)),
    ("payout", re.compile(r"^what happens if", re.I)),
    ("costs", re.compile(r"^what are the costs", re.I)),
    ("holding_period", re.compile(r"^how long should i hold", re.I)),
    ("complaints", re.compile(r"^how can i complain", re.I)),
    ("other_info", re.compile(r"^other relevant information", re.I)),
]
# Sub-headings inside sections: always start a new chunk here, so a table label
# is never separated from its values.
SUBHEADINGS = re.compile(
    r"^(performance scenarios|composition of costs|ongoing costs taken each year|incidental costs taken)",
    re.I,
)
MAX_CHARS = 1500


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    product_name: str
    isin: str | None
    section: str
    page: int


def _detect_section(line, current):
    for name, pattern in SECTION_HEADINGS:
        if pattern.match(line):
            return name
    return current


def _split(lines):
    parts, buffer, size = [], [], 0
    for line in lines:
        if size + len(line) > MAX_CHARS and buffer:
            parts.append("\n".join(buffer))
            buffer, size = [], 0
        buffer.append(line)
        size += len(line) + 1
    if buffer:
        parts.append("\n".join(buffer))
    return parts


def chunk_doc(doc):
    blocks = []
    section, start_page, current = "header", 1, []
    for page in doc.pages:
        for raw in page.text.splitlines():
            line = raw.strip()
            if not line or line.isdigit():  # skip blanks and bare page numbers
                continue
            new_section = _detect_section(line, section)
            if new_section != section or SUBHEADINGS.match(line):
                if current:
                    blocks.append((section, start_page, current))
                section, start_page, current = new_section, page.number, []
            current.append(line)
    if current:
        blocks.append((section, start_page, current))

    chunks = []
    for section, page, lines in blocks:
        for part in _split(lines):
            text = f"[{doc.product_name} | {section}]\n{part}"
            chunk_id = hashlib.sha1(f"{doc.source}:{len(chunks)}".encode()).hexdigest()[:12]
            chunks.append(Chunk(chunk_id, text, doc.source, doc.product_name, doc.isin, section, page))
    return chunks
