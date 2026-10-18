"""Deterministic checks for the multi-source router evaluation (no LLM judge).

Ground truth for client-data questions is computed by an ORACLE SQL query run against the same database, never typed
by hand, so the expected values cannot drift from the data. Known limitations, stated on purpose:
- a number check only asks "does this value appear in the answer"; it does not verify which entity it is attached to;
- small integers (counts) can match by coincidence, so counts are weak evidence on their own.
"""
import re

from wealthai.agent.guardrails import REFUSE_MSG
from wealthai.tools.sql import DEFAULT_DB, run_query

# Ongoing costs printed in the five KIDs of the knowledge base (verified when the golden set was built).
KID_ONGOING = {
    "ishares core msci em imi": "0.18",
    "ishares core msci world": "0.20",
    "ishares core s&p 500": "0.07",
    "vanguard ftse all-world": "0.14",
    "xtrackers msci world": "0.12",
}
_NUMBER = re.compile(r"\d+(?:[,'’]\d{3})*(?:\.\d+)?")
_NOISE = re.compile(r"\b(ucits etf|usd accu|usd acc|etf)\b")


def numbers_in(text: str) -> list[float]:
    """Numbers in a text, with thousands separators (1,494,694 / 1'494'694) removed."""
    return [float(re.sub(r"[,'’]", "", m)) for m in _NUMBER.findall(text)]


def norm(text: str) -> str:
    """Lower-case, drop share-class noise ('UCITS ETF USD Accu') and collapse spaces."""
    return re.sub(r"\s+", " ", _NOISE.sub(" ", text.lower().replace(" ", " "))).strip()


def _oracle_cells(sql: str, db_path) -> tuple[list[float], list[str]]:
    rows = run_query(sql, db_path=db_path).rows
    numbers = [float(c) for row in rows for c in row if isinstance(c, (int, float))]
    texts = [str(c) for row in rows for c in row if isinstance(c, str)]
    return numbers, texts


def _has_number(answer_numbers: list[float], expected: float, tol: float) -> bool:
    return any(abs(x - expected) <= tol for x in answer_numbers)


def check_item(item: dict, result: dict, db_path=DEFAULT_DB) -> dict:
    """Returns {'route_ok', 'answer_ok', 'passed', 'reasons'} for one evaluated question."""
    reasons: list[str] = []
    answer = result["answer"]
    a_norm, a_nums = norm(answer), numbers_in(answer)

    route_ok = result["route"] in item["routes"]
    if not route_ok:
        reasons.append(f"route {result['route']!r} not in {item['routes']}")

    answer_ok = True
    if item.get("must_refuse") and answer != REFUSE_MSG:
        answer_ok = False
        reasons.append("expected the advice refusal")
    if item.get("must_decline") and result["answerable"]:
        answer_ok = False
        reasons.append("expected the system to decline, but it answered")
    for text in item.get("expect_text", []):
        if norm(text) not in a_norm:
            answer_ok = False
            reasons.append(f"missing text {text!r}")
    if item.get("expect_any") and not any(norm(t) in a_norm for t in item["expect_any"]):
        answer_ok = False
        reasons.append(f"none of {item['expect_any']} in the answer")
    for text in item.get("forbid_text", []):
        if norm(text) in a_norm:
            answer_ok = False
            reasons.append(f"forbidden text {text!r} in the answer")

    mode = item.get("expect")
    if mode:
        numbers, texts = _oracle_cells(item["oracle_sql"], db_path)
        if not numbers and not texts:
            answer_ok = False
            reasons.append("oracle returned no rows (bad item)")
        tol = item.get("tol", 0.5)
        if mode in ("numbers", "all"):
            for n in numbers:
                if not _has_number(a_nums, n, tol):
                    answer_ok = False
                    reasons.append(f"missing number {n:g}")
        if mode in ("names", "all", "fund_costs"):
            for t in texts:
                if norm(t) not in a_norm:
                    answer_ok = False
                    reasons.append(f"missing name {t!r}")
        if mode == "fund_costs":
            for t in texts:
                cost = next((c for key, c in KID_ONGOING.items() if key in norm(t)), None)
                if cost and cost not in answer:
                    answer_ok = False
                    reasons.append(f"missing ongoing cost {cost} for {t!r}")

    return {"route_ok": route_ok, "answer_ok": answer_ok, "passed": route_ok and answer_ok, "reasons": reasons}
