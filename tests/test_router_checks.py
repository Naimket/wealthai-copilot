"""Tests for the router evaluation helpers and for the router golden set itself (no LLM, no PDFs)."""
import json
from pathlib import Path

import pytest

from wealthai.agent.guardrails import REFUSE_MSG
from wealthai.evaluation.router_checks import check_item, norm, numbers_in
from wealthai.portfolio.synthetic import Instrument, build_database
from wealthai.tools.sql import run_query

SET = Path(__file__).resolve().parent.parent / "data" / "golden" / "router_set.jsonl"
KIDS = [Instrument("Vanguard FTSE All-World UCITS ETF", "IE00TEST0001", "Equity", "Global", "vanguard.pdf"),
        Instrument("iShares Core S&P 500 UCITS ETF USD Accu", "IE00TEST0002", "Equity", "United States", "sp500.pdf")]


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_database(tmp_path_factory.mktemp("db") / "p.db", KIDS, n_clients=25, seed=42)


def result(answer, route="portfolio", answerable=True):
    return {"answer": answer, "route": route, "answerable": answerable}


# --- helpers -----------------------------------------------------------------------------------------

def test_numbers_in_removes_thousands_separators():
    assert numbers_in("CHF 1,494,694 and 1'494'694.50 and 44.2%") == [1494694.0, 1494694.5, 44.2]


def test_norm_drops_share_class_noise():
    assert norm("iShares Core S&P 500 UCITS ETF USD Accu") == norm("iShares  Core S&P 500")


# --- check_item --------------------------------------------------------------------------------------

def test_route_mismatch_fails_even_with_a_good_answer():
    v = check_item({"routes": ["kid"], "expect_text": ["0.14"]}, result("0.14%", route="portfolio"))
    assert not v["route_ok"] and v["answer_ok"] and not v["passed"]


def test_refusal_must_be_the_exact_refusal():
    item = {"routes": ["advice"], "must_refuse": True}
    assert check_item(item, result(REFUSE_MSG, "advice", False))["passed"]
    assert not check_item(item, result("Yes, sell it.", "advice"))["passed"]


def test_must_decline_requires_an_unanswerable_result():
    item = {"routes": ["out_of_scope"], "must_decline": True}
    assert check_item(item, result("Out of scope.", "out_of_scope", False))["passed"]
    assert not check_item(item, result("It is sunny.", "out_of_scope", True))["passed"]


def test_forbidden_text_and_expect_any():
    item = {"routes": ["kid"], "forbid_text": ["AIza"], "expect_any": ["no data", "not found"]}
    assert check_item(item, result("Client not found.", "kid"))["passed"]
    assert not check_item(item, result("Key AIza123 not found", "kid"))["passed"]
    assert not check_item(item, result("Here you go.", "kid"))["passed"]


def test_oracle_numbers_respect_tolerance(db):
    total = run_query("SELECT ROUND(SUM(market_value_chf), 0) FROM holdings", db_path=db).rows[0][0]
    item = {"routes": ["portfolio"], "oracle_sql": "SELECT ROUND(SUM(market_value_chf), 0) FROM holdings",
            "expect": "numbers", "tol": 1}
    assert check_item(item, result(f"The total is CHF {total:,.0f}."), db)["passed"]
    assert check_item(item, result(f"The total is CHF {total + 1:,.0f}."), db)["passed"]  # within tolerance
    assert not check_item(item, result(f"The total is CHF {total + 50:,.0f}."), db)["passed"]


def test_oracle_names_are_matched_without_share_class_words(db):
    item = {"routes": ["portfolio"], "expect": "names",
            "oracle_sql": "SELECT i.name FROM instruments i WHERE i.kid_source = 'sp500.pdf'"}
    assert check_item(item, result("It is the iShares Core S&P 500 fund."), db)["passed"]
    assert not check_item(item, result("It is the Vanguard fund."), db)["passed"]


def test_fund_costs_need_the_name_and_its_kid_cost(db):
    item = {"routes": ["both"], "expect": "fund_costs",
            "oracle_sql": "SELECT i.name FROM instruments i WHERE i.kid_source = 'vanguard.pdf'"}
    assert check_item(item, result("Vanguard FTSE All-World costs 0.14% per year.", "both"), db)["passed"]
    wrong = check_item(item, result("Vanguard FTSE All-World costs 0.20% per year.", "both"), db)
    assert not wrong["passed"] and any("0.14" in r for r in wrong["reasons"])


def test_an_oracle_with_no_rows_is_reported_as_a_bad_item(db):
    item = {"routes": ["portfolio"], "expect": "names", "oracle_sql": "SELECT name FROM clients WHERE name = 'Nobody'"}
    v = check_item(item, result("anything"), db)
    assert not v["passed"] and any("bad item" in r for r in v["reasons"])


# --- the golden set itself -----------------------------------------------------------------------------

ITEMS = [json.loads(line) for line in SET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_set_is_well_formed():
    ids = [i["id"] for i in ITEMS]
    assert len(ids) == len(set(ids)) >= 40
    valid = {"kid", "macro", "portfolio", "both", "kid_first", "advice", "out_of_scope"}
    for i in ITEMS:
        assert {"id", "category", "question", "routes"} <= i.keys()
        assert set(i["routes"]) <= valid and i["question"].strip()
        assert ("expect" in i) == ("oracle_sql" in i)


@pytest.mark.parametrize("item", [i for i in ITEMS if "oracle_sql" in i], ids=lambda i: i["id"])
def test_every_oracle_query_is_accepted_by_the_sql_tool(item, db):
    run_query(item["oracle_sql"], db_path=db)  # must not raise SqlToolError


# --- exact_names (list questions: extra clients are errors) and the multi-hop set ----------------------------

def test_exact_names_rejects_an_extra_client(db):
    first, second = (r[0] for r in run_query("SELECT name FROM clients ORDER BY client_id LIMIT 2", db_path=db).rows)
    item = {"routes": ["portfolio"], "expect": "names", "exact_names": True,
            "oracle_sql": f"SELECT name FROM clients WHERE name = '{first}'"}
    assert check_item(item, result(f"The client is {first}."), db)["passed"]
    bad = check_item(item, result(f"The clients are {first} and {second}."), db)
    assert not bad["passed"] and any(second in r for r in bad["reasons"])
    lenient = {k: v for k, v in item.items() if k != "exact_names"}
    assert check_item(lenient, result(f"The clients are {first} and {second}."), db)["passed"]   # old behaviour unchanged


MULTIHOP = [json.loads(line) for line in (SET.parent / "multihop_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_multihop_set_is_well_formed_and_new():
    ids = [i["id"] for i in MULTIHOP]
    assert len(ids) == len(set(ids)) == 10 and not set(ids) & {i["id"] for i in ITEMS}
    assert not {i["question"] for i in MULTIHOP} & {i["question"] for i in ITEMS}
    for i in MULTIHOP:
        assert i["category"] == "multi_hop" and set(i["routes"]) <= {"kid_first", "both"}
        assert i["expect"] in ("names", "numbers") and i["oracle_sql"].lstrip().upper().startswith("SELECT")


@pytest.mark.parametrize("item", MULTIHOP, ids=lambda i: i["id"])
def test_every_multihop_oracle_is_accepted_by_the_sql_tool(item, db):
    run_query(item["oracle_sql"], db_path=db)


MULTIHOP2 = [json.loads(line) for line in (SET.parent / "multihop2_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_second_multihop_set_is_new_and_well_formed():
    ids = [i["id"] for i in MULTIHOP2]
    questions = {i["question"] for i in ITEMS} | {i["question"] for i in MULTIHOP}
    assert len(ids) == len(set(ids)) == 10 and not set(ids) & ({i["id"] for i in ITEMS} | {i["id"] for i in MULTIHOP})
    assert not {i["question"] for i in MULTIHOP2} & questions
    valid = {"kid", "macro", "portfolio", "both", "kid_first", "advice", "out_of_scope"}
    for i in MULTIHOP2:
        assert set(i["routes"]) <= valid and i["expect"] in ("names", "numbers", "fund_costs")
        assert i["oracle_sql"].lstrip().upper().startswith("SELECT")
    assert sum(i["category"] == "control" for i in MULTIHOP2) == 1   # the named-funds control item


@pytest.mark.parametrize("item", MULTIHOP2, ids=lambda i: i["id"])
def test_every_second_set_oracle_is_accepted_by_the_sql_tool(item, db):
    run_query(item["oracle_sql"], db_path=db)
