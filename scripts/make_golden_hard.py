"""Append a HARD block to the golden set: paraphrases, nicknames, aggregation, ambiguity, near-miss traps.

Run after make_golden.py (it re-reads golden_set.jsonl, drops any previous hard block, appends a new one).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.ingestion.parse import parse_pdf  # noqa: E402

SRI = re.compile(r"classified (?:this|the) (?:product|fund)\s+as\s+(\d)\s+out of 7", re.I)
ONGOING = re.compile(r"Management fees.{0,160}?(\d+[.,]\d+)\s?%", re.I | re.S)
TRANSACTION = re.compile(r"Transaction costs\s.{0,80}?(\d+[.,]\d+)\s?%", re.I | re.S)

NICK = {  # how an advisor would casually call each fund + a token that identifies it in an answer
    "ishares_core_msci_em_imi.pdf": ("iShares emerging markets IMI ETF", "EM IMI"),
    "ishares_core_msci_world.pdf": ("iShares Core World ETF", "iShares Core MSCI World"),
    "ishares_core_s&p_500.pdf": ("iShares S&P 500 ETF", "S&P 500"),
    "vanguard_ftse_all_world.pdf": ("Vanguard All-World ETF", "Vanguard"),
    "xtrackers_msci_world.pdf": ("Xtrackers World ETF", "Xtrackers"),
}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def find(pattern, text):
    m = pattern.search(re.sub(r"\s+", " ", text))
    return m.group(1).replace(",", ".") if m else None


def pct(value):
    num = float(value)
    forms = {f"{num:.2f}", f"{num:.1f}", f"{num:g}"}
    return r"(?:" + "|".join(re.escape(f).replace(r"\.", "[.,]") for f in sorted(forms, key=len, reverse=True)) + r")\s?%"


def all_of(*patterns):  # every pattern must appear (any order)
    return "(?s)^" + "".join(f"(?=.*(?:{p}))" for p in patterns)


def in_order(*patterns):  # patterns must appear in this order
    return "(?s)" + ".*".join(f"(?:{p})" for p in patterns)


def build(funds):
    items = []

    def add(question, match, answerable=True, reference="", source=None):
        items.append({"type": "hard", "question": question, "source": source, "match": match,
                      "answerable": answerable, "reference": reference})

    for f in funds:
        nick, _ = NICK.get(f["source"], (f["name"], f["name"]))
        if f["ongoing"]:
            add(f"What is the TER (expense ratio) of the {f['name']}?", [pct(f["ongoing"])],
                reference=f"{f['ongoing']}% (paraphrase: TER)", source=f["source"])
            add(f"What does the {nick} charge per year in ongoing fees?", [pct(f["ongoing"])],
                reference=f"{f['ongoing']}% (nickname)", source=f["source"])
        if f["sri"]:
            add(f"On the 1-to-7 risk scale, how risky is the {f['name']}?",
                [rf"\b{f['sri']}\b\s*(?:out of|of|/)\s*7"], reference=f"{f['sri']} out of 7 (paraphrase)", source=f["source"])

    with_cost = [f for f in funds if f["ongoing"]]
    if len(with_cost) >= 3:
        ranked = sorted(with_cost, key=lambda f: float(f["ongoing"]))
        tokens = [NICK.get(f["source"], (f["name"], re.escape(f["name"])))[1] for f in ranked]
        add("Rank all the funds from the cheapest to the most expensive in ongoing costs.",
            [in_order(*tokens)], reference=" < ".join(f"{f['name']} {f['ongoing']}%" for f in ranked))
        below = [f for f in with_cost if float(f["ongoing"]) < 0.15]
        if below:
            add("Which funds have ongoing costs below 0.15% per year?",
                [all_of(*[NICK.get(f["source"], (f["name"], re.escape(f["name"])))[1] for f in below])],
                reference=", ".join(f"{f['name']} ({f['ongoing']}%)" for f in below))
    zero = [f for f in funds if f["transaction"] is not None and float(f["transaction"]) == 0]
    if len(funds) >= 3 and zero and len(zero) in WORDS:
        n = len(zero)
        add("How many of the funds report transaction costs of 0.00% per year?",
            [rf"\b{n}\b|\b{WORDS[n]}\b"], reference=f"{n}: " + ", ".join(f["name"] for f in zero))
    worlds = [f for f in funds if "world" in f["name"].lower() and "all-world" not in f["name"].lower() and f["ongoing"]]
    if len(worlds) >= 2:
        add("What are the ongoing costs of the MSCI World ETF?",
            [all_of(*[pct(f["ongoing"]) for f in worlds])],
            reference="Ambiguous: " + " and ".join(f"{f['name']} {f['ongoing']}%" for f in worlds))

    for q in [  # near-miss traps: similar entity or a figure the KID does not contain
        "What are the ongoing costs of the Amundi MSCI World UCITS ETF?",
        "What is the ISIN of the Vanguard S&P 500 UCITS ETF?",
        "What dividend yield does the Vanguard FTSE All-World UCITS ETF currently pay?",
        "What are the ongoing costs of the iShares Core MSCI Europe UCITS ETF?",
    ]:
        add(q, None, answerable=False, reference="Not in the documents (near-miss trap)")
    return items


if __name__ == "__main__":
    funds = []
    for pdf in sorted((ROOT / "data" / "raw").glob("*.pdf")):
        doc = parse_pdf(pdf)
        text = "\n".join(p.text for p in doc.pages)
        funds.append({"source": pdf.name, "name": doc.product_name, "sri": find(SRI, text),
                      "ongoing": find(ONGOING, text), "transaction": find(TRANSACTION, text)})

    path = ROOT / "data" / "golden" / "golden_set.jsonl"
    base = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    base = [i for i in base if i["type"] != "hard"]
    for item in build(funds):
        item["id"] = f"q{len(base) + 1:02d}"
        base.append(item)
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in base) + "\n", encoding="utf-8")
    print(f"Golden set now has {len(base)} questions ({sum(i['type'] == 'hard' for i in base)} hard)")
