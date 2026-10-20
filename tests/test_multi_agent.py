"""Control-flow tests for the multi-source agent. The LLM and the KID agent are replaced by scripted fakes,
so these tests check the routing, the SQL retry, the verification loop and the guardrails: no API, no models."""
import pytest

from wealthai.agent import multi
from wealthai.agent.guardrails import ABSTAIN_MSG, REFUSE_MSG
from wealthai.portfolio.synthetic import Instrument, build_database

GOOD_SQL = "SELECT COUNT(*) AS n FROM clients"
KID_RESULT = {
    "answer": "Test World ETF has ongoing costs of 0.20% per year.",
    "answerable": True,
    "citations": [1],
    "contexts": ["Test World ETF | costs: ongoing costs 0.20%"],
    "sources": ["test.pdf"],
    "sections": ["What are the costs?"],
    "trace": ["guard", "rewrite", "retrieve", "grade", "generate", "verify"],
}


MACRO_RESULT = {
    "answer": "The SNB left the policy rate at 0% on 24 September 2026.",
    "answerable": True,
    "citations": [1],
    "contexts": ["[SNB Monetary policy assessment of 24 September 2026 | PDF p.1] policy rate 0%"],
    "sources": ["snb_monetary_policy_assessment_2026_09_24.pdf"],
    "sections": ["SNB Monetary policy assessment of 24 September 2026"],
    "trace": ["guard", "rewrite", "retrieve", "grade", "generate", "verify"],
}


class FakeLLM:
    """Returns the scripted objects for each schema, in order; the last one repeats."""

    def __init__(self, **by_schema):
        self.queues = {name: list(items) for name, items in by_schema.items()}
        self.calls = []
        self.prompts = []

    def __call__(self, prompt, schema=None, **kwargs):
        self.calls.append(schema.__name__)
        self.prompts.append((schema.__name__, prompt))
        queue = self.queues[schema.__name__]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def count(self, schema_name):
        return self.calls.count(schema_name)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    db = build_database(tmp_path / "p.db", [Instrument("Test World ETF", "IE00TEST0001", "Equity", "Global", "test.pdf")],
                        n_clients=10, seed=1)
    monkeypatch.setattr(multi, "DB_PATH", db)
    monkeypatch.setattr(multi, "_graph", None)
    kid_calls = []

    def fake_kid(question):
        kid_calls.append(question)
        return dict(KID_RESULT)

    monkeypatch.setattr(multi, "run_kid_agent", fake_kid)
    macro_calls = []

    def fake_macro(question):
        macro_calls.append(question)
        return dict(MACRO_RESULT)

    monkeypatch.setattr(multi, "run_macro_agent", fake_macro)
    monkeypatch.setattr(multi, "macro_calls", macro_calls, raising=False)

    def install(**scripts):
        llm = FakeLLM(**scripts)
        monkeypatch.setattr(multi, "generate", llm)
        return llm

    return install, kid_calls


def R(route):
    return multi.Route(route=route, reason="test")


def FA(text="ok", answerable=True):
    return multi.FinalAnswer(answerable=answerable, answer=text)


def V(supported, unsupported=""):
    return multi.Verification(supported=supported, unsupported=unsupported)


# --- guardrails and trivial routes ------------------------------------------------------------------

def test_regex_guard_refuses_without_any_llm_call(setup):
    install, kid_calls = setup
    llm = install(Route=[R("kid")])
    out = multi.run_multi("Should I buy the Vanguard fund for Alice?")
    assert out["answer"] == REFUSE_MSG and not out["answerable"] and out["route"] == "advice"
    assert llm.calls == [] and kid_calls == []


def test_llm_router_can_refuse_suitability_questions(setup):
    install, kid_calls = setup
    llm = install(Route=[R("advice")])
    out = multi.run_multi("Is the Xtrackers fund suitable for Alice Brunner?")
    assert out["answer"] == REFUSE_MSG and out["route"] == "advice"
    assert llm.calls == ["Route"] and kid_calls == []


def test_out_of_scope(setup):
    install, kid_calls = setup
    install(Route=[R("out_of_scope")])
    out = multi.run_multi("What is the weather in Nyon?")
    assert out["answer"] == multi.OUT_OF_SCOPE_MSG and kid_calls == []


def test_unknown_route_falls_back_to_kid_agent(setup):
    install, kid_calls = setup
    install(Route=[R("banana")])
    out = multi.run_multi("What are the costs of the Test World ETF?")
    assert out["route"] == "kid" and len(kid_calls) == 1


# --- KID-only route ---------------------------------------------------------------------------------

def test_kid_route_passes_the_question_through_and_returns_its_answer(setup):
    install, kid_calls = setup
    llm = install(Route=[R("kid")])
    out = multi.run_multi("What are the ongoing costs of the Test World ETF?")
    assert kid_calls == ["What are the ongoing costs of the Test World ETF?"]
    assert out["answer"] == KID_RESULT["answer"] and out["answerable"]
    assert out["citations"] == [1] and out["sql"] == ""
    assert llm.calls == ["Route"]  # no SQL, no compose, no verify
    assert any("kid_agent" in step for step in out["trace"])


# --- macro route ------------------------------------------------------------------------------------

def test_macro_route_passes_the_question_through_and_returns_its_answer(setup):
    install, kid_calls = setup
    llm = install(Route=[R("macro")])
    out = multi.run_multi("What did the SNB decide on 24 September 2026?")
    assert multi.macro_calls == ["What did the SNB decide on 24 September 2026?"]
    assert out["route"] == "macro" and out["answer"] == MACRO_RESULT["answer"] and out["answerable"]
    assert out["citations"] == [1] and out["sql"] == ""
    assert out["sources"] == MACRO_RESULT["sources"] and out["contexts"] == MACRO_RESULT["contexts"]
    assert kid_calls == [] and llm.calls == ["Route"]  # no SQL, no compose, no KID agent
    assert any("macro_agent" in step for step in out["trace"])


def test_macro_route_keeps_an_honest_abstention(setup, monkeypatch):
    install, _ = setup
    install(Route=[R("macro")])
    abstained = dict(MACRO_RESULT, answer=ABSTAIN_MSG, answerable=False, citations=[], contexts=[], sources=[])
    monkeypatch.setattr(multi, "run_macro_agent", lambda question: abstained)
    out = multi.run_multi("What will the SNB policy rate be in 2030?")
    assert out["route"] == "macro" and out["answer"] == ABSTAIN_MSG and not out["answerable"]


def test_advice_about_macro_is_still_refused_before_routing(setup):
    install, _ = setup
    llm = install(Route=[R("macro")])
    out = multi.run_multi("Should I buy Swiss bonds given the latest SNB decision?")
    assert out["answer"] == REFUSE_MSG and llm.calls == [] and multi.macro_calls == []


# --- portfolio-only route ---------------------------------------------------------------------------

def test_portfolio_route_runs_sql_composes_and_verifies(setup):
    install, kid_calls = setup
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("There are 10 clients.")], Verification=[V(True)])
    out = multi.run_multi("How many clients do we have?")
    assert out["answer"] == "There are 10 clients." and out["answerable"]
    assert out["sql"] == GOOD_SQL and kid_calls == []
    assert llm.calls == ["Route", "SqlQuery", "FinalAnswer", "Verification"]
    assert out["contexts"][-1].startswith("SQL: SELECT COUNT(*)") and "10" in out["contexts"][-1]
    assert out["sources"][-1] == "portfolios.db"


def test_sql_error_is_fed_back_and_retried_once(setup):
    install, _ = setup
    llm = install(Route=[R("portfolio")],
                  SqlQuery=[multi.SqlQuery(sql="DELETE FROM clients"), multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("10 clients.")], Verification=[V(True)])
    out = multi.run_multi("How many clients do we have?")
    assert llm.count("SqlQuery") == 2 and out["answer"] == "10 clients."
    assert any(step.startswith("sql_run error") for step in out["trace"])


def test_two_sql_failures_give_a_safe_message(setup):
    install, _ = setup
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql="DROP TABLE clients")])
    out = multi.run_multi("Delete everything")
    assert out["answer"] == multi.SQL_FAILED_MSG and not out["answerable"]
    assert llm.count("SqlQuery") == multi.MAX_SQL_ATTEMPTS and llm.count("FinalAnswer") == 0


def test_empty_result_still_goes_through_compose(setup):
    install, _ = setup
    install(Route=[R("portfolio")],
            SqlQuery=[multi.SqlQuery(sql="SELECT name FROM clients WHERE name = 'Nobody'")],
            FinalAnswer=[FA("No matching client data was found.")], Verification=[V(True)])
    out = multi.run_multi("What does Nobody hold?")
    assert "(no rows)" in out["contexts"][-1] and "No matching" in out["answer"]


# --- both sources -----------------------------------------------------------------------------------

def test_both_route_asks_the_kid_agent_a_self_contained_question(setup):
    install, kid_calls = setup
    llm = install(Route=[R("both")], SqlQuery=[multi.SqlQuery(sql="SELECT name, kid_source FROM instruments")],
                  KidQuestion=[multi.KidQuestion(needed=True, question="What are the ongoing costs of Test World ETF?")],
                  FinalAnswer=[FA("Alice holds Test World ETF (0.20% ongoing costs).")], Verification=[V(True)])
    out = multi.run_multi("What do the funds Alice holds cost?")
    assert kid_calls == ["What are the ongoing costs of Test World ETF?"]
    assert llm.calls == ["Route", "SqlQuery", "KidQuestion", "FinalAnswer", "Verification"]
    assert out["answerable"] and any(c.startswith("SQL:") for c in out["contexts"])
    assert KID_RESULT["contexts"][0] in out["contexts"]  # KID passages travel with the answer


def test_both_route_skips_the_kid_agent_when_no_instrument_has_a_kid(setup):
    install, kid_calls = setup
    llm = install(Route=[R("both")], SqlQuery=[multi.SqlQuery(sql="SELECT name FROM instruments WHERE kid_source IS NULL")],
                  KidQuestion=[multi.KidQuestion(needed=False)],
                  FinalAnswer=[FA("No KID is available for these instruments.")], Verification=[V(True)])
    out = multi.run_multi("What do the bond funds Alice holds cost?")
    assert kid_calls == [] and llm.count("KidQuestion") == 1 and out["answerable"]


# --- verification loop ------------------------------------------------------------------------------

def test_unsupported_draft_is_regenerated_once(setup):
    install, _ = setup
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("There are 11 clients."), FA("There are 10 clients.")],
                  Verification=[V(False, "11 is not in the evidence"), V(True)])
    out = multi.run_multi("How many clients do we have?")
    assert out["answer"] == "There are 10 clients." and llm.count("FinalAnswer") == 2


def test_still_unsupported_after_one_retry_abstains(setup):
    install, _ = setup
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("There are 11 clients.")], Verification=[V(False, "made up")])
    out = multi.run_multi("How many clients do we have?")
    assert out["answer"] == ABSTAIN_MSG and not out["answerable"] and llm.count("FinalAnswer") == 2


def test_unanswerable_compose_skips_verification(setup):
    install, _ = setup
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("The data cannot answer this.", answerable=False)])
    out = multi.run_multi("What is the meaning of life?")
    assert not out["answerable"] and llm.count("Verification") == 0


# --- the redactor and the verifier must see WHAT the query selected (regression: a bare "1" is not "clients in Geneva") ---

def test_compose_and_verify_prompts_include_the_sql_that_produced_the_rows(setup):
    install, _ = setup
    sql = "SELECT COUNT(*) AS n FROM clients WHERE city = 'Geneva'"
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=sql)],
                  FinalAnswer=[FA("1 client lives in Geneva.")], Verification=[V(True)])
    multi.run_multi("How many clients do we have in Geneva?")
    prompts = dict(llm.prompts)
    assert sql in prompts["FinalAnswer"] and sql in prompts["Verification"]


def test_sql_prompt_states_the_rounding_limit_and_missing_columns_rules(setup):
    install, _ = setup
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("10 clients.")], Verification=[V(True)])
    multi.run_multi("How many clients do we have?")
    sql_prompt = dict(llm.prompts)["SqlQuery"]
    assert "never round counts or ages" in sql_prompt
    assert "only when the question asks for a top-N" in sql_prompt
    assert "NO cost, fee or risk columns" in sql_prompt


# --- KID-first route ----------------------------------------------------------------------------------

def KQ(question="What are the ongoing costs of each of the five funds?"):
    return multi.KidQuestion(needed=True, question=question)


def test_kid_first_asks_the_kids_then_filters_the_database_by_their_facts(setup):
    install, kid_calls = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("10 clients hold it.")], Verification=[V(True)])
    out = multi.run_multi("Which clients hold the fund with the lowest ongoing costs?")
    assert out["route"] == "kid_first" and out["answer"] == "10 clients hold it."
    assert llm.calls == ["Route", "KidQuestion", "SqlQuery", "FinalAnswer", "Verification"]
    assert kid_calls == ["What are the ongoing costs of each of the five funds?"]  # the KID agent sees the generated question
    sql_prompt = [p for s, p in llm.prompts if s == "SqlQuery"][0]
    assert "KID FACTS" in sql_prompt and KID_RESULT["answer"] in sql_prompt  # the SQL step sees the KID answer
    compose_prompt = [p for s, p in llm.prompts if s == "FinalAnswer"][0]
    assert KID_RESULT["answer"] in compose_prompt
    assert out["sources"][-1] == "portfolios.db" and "test.pdf" in out["sources"]


def test_kid_first_falls_back_to_the_advisor_question_when_none_is_generated(setup):
    install, kid_calls = setup
    install(Route=[R("kid_first")], KidQuestion=[multi.KidQuestion(needed=True, question="  ")],
            SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)], FinalAnswer=[FA()], Verification=[V(True)])
    multi.run_multi("Which clients hold a fund with ongoing costs above 0.15%?")
    assert kid_calls == ["Which clients hold a fund with ongoing costs above 0.15%?"]


def test_kid_first_abstains_without_querying_when_the_kids_cannot_answer(setup, monkeypatch):
    install, _ = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)])
    monkeypatch.setattr(multi, "run_kid_agent", lambda q: dict(KID_RESULT, answerable=False, answer=ABSTAIN_MSG))
    out = multi.run_multi("Which clients hold the fund with the lowest tracking error?")
    assert out["answer"] == ABSTAIN_MSG and not out["answerable"]
    assert "SqlQuery" not in llm.calls


def test_kid_first_still_refuses_advice_before_any_call(setup):
    install, kid_calls = setup
    llm = install(Route=[R("kid_first")])
    out = multi.run_multi("Which fund should Alice buy, the one with the lowest ongoing costs?")
    assert out["answer"] == REFUSE_MSG and llm.calls == [] and kid_calls == []
