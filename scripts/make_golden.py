"""Build the golden set from the KIDs in data/raw (facts are regex-extracted, you verify them)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.ingestion.parse import parse_pdf  # noqa: E402

SRI = re.compile(r"classified (?:this|the) (?:product|fund)\s+as\s+(\d)\s+out of 7", re.I)
HOLD = re.compile(r"Recommended holding period\s*:?\s*(\d+)\s+years", re.I)
ONGOING = re.compile(r"Management fees.{0,160}?(\d+[.,]\d+)\s?%", re.I | re.S)
TRANSACTION = re.compile(r"Transaction costs\s.{0,80}?(\d+[.,]\d+)\s?%", re.I | re.S)


def find(pattern, text):
    m = pattern.search(re.sub(r"\s+", " ", text))
    return m.group(1).replace(",", ".") if m else None


def pct(value):  # '0.20' -> regex that matches 0.20%, 0.2 %, 0,20%
    num = float(value)
    forms = {f"{num:.2f}", f"{num:.1f}", f"{num:g}"}
    return r"(?:" + "|".join(re.escape(f).replace(r"\.", "[.,]") for f in sorted(forms, key=len, reverse=True)) + r")\s?%"


funds = []
for pdf in sorted((ROOT / "data" / "raw").glob("*.pdf")):
    doc = parse_pdf(pdf)
    text = "\n".join(p.text for p in doc.pages)
    funds.append({
        "source": pdf.name, "name": doc.product_name, "isin": doc.isin,
        "sri": find(SRI, text), "holding": find(HOLD, text),
        "ongoing": find(ONGOING, text), "transaction": find(TRANSACTION, text),
    })

print(f"{'fund':55s} {'ISIN':14s} SRI hold ongoing transaction")
for f in funds:
    print(f"{f['name'][:55]:55s} {f['isin']:14s} {f['sri']}   {f['holding']}    {f['ongoing']}     {f['transaction']}")

items = []
def add(qtype, question, source=None, match=None, answerable=True, reference=""):
    items.append({"id": f"q{len(items) + 1:02d}", "type": qtype, "question": question,
                  "source": source, "match": match, "answerable": answerable, "reference": reference})

for f in funds:
    n = f["name"]
    if f["sri"]:
        add("factual", f"What is the summary risk indicator (SRI) of the {n}?", f["source"],
            [rf"\b{f['sri']}\b\s*(?:out of|of|/)\s*7"], reference=f"{f['sri']} out of 7")
    if f["ongoing"]:
        add("factual", f"What are the ongoing management fees and other administrative or operating costs per year of the {n}?",
            f["source"], [pct(f["ongoing"])], reference=f"{f['ongoing']}% per year")
    if f["holding"]:
        add("factual", f"What is the recommended holding period of the {n}?", f["source"],
            [rf"\b{f['holding']}\s*(?:years|year|-year)"], reference=f"{f['holding']} years")
    if f["transaction"]:
        add("factual", f"What are the transaction costs per year of the {n}?", f["source"],
            [pct(f["transaction"])], reference=f"{f['transaction']}% per year")

ranked = sorted((f for f in funds if f["ongoing"]), key=lambda f: float(f["ongoing"]))
if len(ranked) >= 2:
    cheap, dear = ranked[0], ranked[-1]
    add("comparative", "Which of the funds has the lowest ongoing management fees and other administrative or operating costs?",
        None, [re.escape(cheap["name"].split()[0])], reference=f"{cheap['name']} ({cheap['ongoing']}%)")
    add("comparative", "Which of the funds has the highest ongoing management fees and other administrative or operating costs?",
        None, [re.escape(dear["name"].split()[0])], reference=f"{dear['name']} ({dear['ongoing']}%)")
    a, b = ranked[0], ranked[1]
    add("comparative", f"Is the {a['name']} cheaper than the {b['name']} in ongoing costs?",
        None, [r"\byes\b|cheaper|lower"], reference=f"Yes: {a['ongoing']}% vs {b['ongoing']}%")
sri_vals = {f["sri"] for f in funds if f["sri"]}
add("comparative", "Do all the funds have the same summary risk indicator?", None,
    [r"\byes\b|same"] if len(sri_vals) == 1 else [r"\bno\b|differ"],
    reference="Yes" if len(sri_vals) == 1 else "No")

for q in [
    "What was the 10-year annualised past performance of the Vanguard FTSE All-World UCITS ETF?",
    "Who is the portfolio manager of the iShares Core MSCI World UCITS ETF?",
    "What is the current share price of the Xtrackers MSCI World UCITS ETF?",
    "What is the current policy rate of the Swiss National Bank?",
    "Should my client who is 60 years old and risk-averse buy the iShares Core S&P 500 UCITS ETF?",
    "What is the tracking error of the iShares Core MSCI EM IMI UCITS ETF last year?",
    "What is the total assets under management of the Vanguard FTSE All-World UCITS ETF?",
    "What was the Swiss inflation rate last month?",
]:
    add("unanswerable", q, None, None, answerable=False, reference="Documents do not contain it / no advice")

out = ROOT / "data" / "golden" / "golden_set.jsonl"
out.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
print(f"\nSaved {len(items)} questions to {out}")
