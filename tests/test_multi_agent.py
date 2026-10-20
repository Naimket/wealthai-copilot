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


def SEL(*names):
    return multi.InstrumentSelection(names=list(names), reason="test")


def test_kid_first_asks_the_kids_selects_instruments_then_filters_the_database(setup):
    install, kid_calls = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()], InstrumentSelection=[SEL("Test World ETF")],
                  SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)], FinalAnswer=[FA("10 clients hold it.")],
                  Verification=[V(True)])
    out = multi.run_multi("Which clients hold the fund with the lowest ongoing costs?")
    assert out["route"] == "kid_first" and out["answer"] == "10 clients hold it."
    assert llm.calls == ["Route", "KidQuestion", "InstrumentSelection", "SqlQuery", "FinalAnswer", "Verification"]
    assert kid_calls == ["What are the ongoing costs of each of the five funds?"]  # the KID agent sees the generated question
    sql_prompt = [p for s, p in llm.prompts if s == "SqlQuery"][0]
    assert KID_RESULT["answer"] in sql_prompt and "instruments.isin IN ('IE00TEST0001')" in sql_prompt
    compose_prompt = [p for s, p in llm.prompts if s == "FinalAnswer"][0]
    assert KID_RESULT["answer"] in compose_prompt
    assert out["sources"][-1] == "portfolios.db" and "test.pdf" in out["sources"]


def test_selected_names_are_validated_against_the_database(setup):
    install, _ = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()],
                  InstrumentSelection=[SEL("test world etf", "Imaginary Fund ETF", "Test World ETF")],
                  SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)], FinalAnswer=[FA()], Verification=[V(True)])
    out = multi.run_multi("Which clients hold the cheapest fund?")
    sql_prompt = [p for s, p in llm.prompts if s == "SqlQuery"][0]
    assert "Imaginary" not in sql_prompt                      # an invented name never reaches the SQL step
    assert sql_prompt.count("IE00TEST0001") == 1              # case-insensitive match, no duplicates
    assert any("select_instruments" in step for step in out["trace"])


def test_kid_first_abstains_when_no_instrument_qualifies(setup):
    install, _ = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()], InstrumentSelection=[SEL("Imaginary Fund ETF")],
                  SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)])
    out = multi.run_multi("Which clients hold the fund with the lowest ongoing costs?")
    assert out["answer"] == ABSTAIN_MSG and not out["answerable"]
    assert "SqlQuery" not in llm.calls


def test_kid_first_falls_back_to_the_advisor_question_when_none_is_generated(setup):
    install, kid_calls = setup
    install(Route=[R("kid_first")], KidQuestion=[multi.KidQuestion(needed=True, question="  ")],
            InstrumentSelection=[SEL("Test World ETF")], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)],
            FinalAnswer=[FA()], Verification=[V(True)])
    multi.run_multi("Which clients hold a fund with ongoing costs above 0.15%?")
    assert kid_calls == ["Which clients hold a fund with ongoing costs above 0.15%?"]


def test_kid_first_abstains_without_querying_when_the_kids_cannot_answer(setup, monkeypatch):
    install, _ = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()], InstrumentSelection=[SEL("Test World ETF")],
                  SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)])
    monkeypatch.setattr(multi, "run_kid_agent", lambda q: dict(KID_RESULT, answerable=False, answer=ABSTAIN_MSG))
    out = multi.run_multi("Which clients hold the fund with the lowest tracking error?")
    assert out["answer"] == ABSTAIN_MSG and not out["answerable"]
    assert "SqlQuery" not in llm.calls and "InstrumentSelection" not in llm.calls


def test_kid_first_still_refuses_advice_before_any_call(setup):
    install, kid_calls = setup
    llm = install(Route=[R("kid_first")])
    out = multi.run_multi("Which fund should Alice buy, the one with the lowest ongoing costs?")
    assert out["answer"] == REFUSE_MSG and llm.calls == [] and kid_calls == []


def test_listing_questions_get_a_completeness_rule_in_compose_and_verify():
    assert "never shorten the list" in multi.COMPOSE_PROMPT
    assert "leaves out any" in multi.VERIFY_PROMPT


def test_qualifying_funds_reach_compose_and_verify_on_the_kid_first_route_only(setup):
    install, _ = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()], InstrumentSelection=[SEL("Test World ETF")],
                  SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)], FinalAnswer=[FA()], Verification=[V(True)])
    multi.run_multi("Which clients hold the fund with the lowest ongoing costs?")
    for schema in ("FinalAnswer", "Verification"):
        prompt = [p for s, p in llm.prompts if s == schema][0]
        assert "chosen from the KID ANSWER by comparing" in prompt and "holding them): Test World ETF" in prompt


def test_other_routes_have_no_qualifying_block(setup):
    install, _ = setup
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)], FinalAnswer=[FA()],
                  Verification=[V(True)])
    multi.run_multi("How many clients do we have?")
    assert all("chosen from the KID ANSWER by comparing" not in p for s, p in llm.prompts if s in ("FinalAnswer", "Verification"))


# --- fixes after the unseen multi-hop run (n03, n04, n08, n09, n10) -----------------------------------------

def test_route_schema_lists_every_route():
    """n03 and n08 were declined because the field description omitted kid_first ('not in the allowed route list')."""
    description = multi.Route.model_fields["route"].description
    assert all(route in description for route in multi.ROUTES)


def test_route_prompt_describes_kid_first_phrasings():
    for phrase in ("second-highest", "exactly 0.12%", "both extremes", "none of the funds"):
        assert phrase in multi.ROUTE_PROMPT
    assert "which\n  holdings have a KID in the knowledge base" in multi.ROUTE_PROMPT   # r25: the portfolio route must stay reachable for it


def test_sql_filtering_a_text_column_by_a_number_is_rejected_and_retried(setup):
    install, _ = setup
    invented = "SELECT c.name FROM clients c JOIN holdings h ON c.client_id = h.client_id JOIN instruments i ON h.instrument_id = i.instrument_id WHERE i.kid_source LIKE '%0.12%'"
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=invented), multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("10 clients.")], Verification=[V(True)])
    out = multi.run_multi("List the clients who hold a fund with ongoing costs of exactly 0.12%.")
    assert llm.count("SqlQuery") == 2 and out["sql"] == GOOD_SQL
    assert any("filters a text column by a number" in step for step in out["trace"])


def test_two_invented_numeric_filters_end_in_a_safe_message_not_a_confident_empty_answer(setup):
    install, _ = setup
    invented = "SELECT name FROM instruments WHERE lower(name) LIKE lower('%0.07%')"
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=invented)])
    out = multi.run_multi("Which clients hold a fund with ongoing costs of 0.07%?")
    assert out["answer"] == multi.SQL_FAILED_MSG and not out["answerable"] and llm.count("FinalAnswer") == 0


def test_numeric_like_guard_leaves_normal_name_filters_alone():
    assert not multi._NUMERIC_LIKE.search("SELECT 1 FROM instruments WHERE name LIKE '%S&P 500%'")
    assert not multi._NUMERIC_LIKE.search("SELECT 1 FROM clients WHERE lower(clients.name) LIKE lower('%alice%')")
    assert multi._NUMERIC_LIKE.search("SELECT 1 FROM instruments WHERE kid_source LIKE '%0.12%'")
    assert multi._NUMERIC_LIKE.search("SELECT 1 FROM instruments WHERE lower(name) LIKE lower('%0.07%')")


def test_selection_and_sql_prompts_carry_the_quantifier_rules(setup):
    install, _ = setup
    llm = install(Route=[R("kid_first")], KidQuestion=[KQ()], InstrumentSelection=[SEL("Test World ETF")],
                  SqlQuery=[multi.SqlQuery(sql=GOOD_SQL)], FinalAnswer=[FA()], Verification=[V(True)])
    multi.run_multi("How many clients hold none of the funds with ongoing costs above 0.15%?")
    select_prompt = [p for s, p in llm.prompts if s == "InstrumentSelection"][0]
    sql_prompt = [p for s, p in llm.prompts if s == "SqlQuery"][0]
    assert "never select the complement" in select_prompt
    assert "HAVING COUNT(DISTINCT instruments.isin)" in sql_prompt and "NOT IN (SELECT holdings.client_id" in sql_prompt


def test_compose_prompt_does_not_let_the_ranking_rule_override_the_qualifying_funds():
    """n03 and n08: correct SQL, then 'the available data does not allow that conclusion' from the generic ranking rule."""
    prompt = multi.COMPOSE_PROMPT
    assert "second-highest cost, both\n  extremes" in prompt and "do not say the KID ANSWER fails to rank the funds" in prompt
    assert "(this does not apply when FUNDS THAT QUALIFY is given: the ranking is then established)" in prompt


def test_empty_result_from_an_exact_instrument_name_is_retried_with_the_error(setup):
    """p10: `i1.name = 'Vanguard FTSE All-World'` matched nothing and the answer was a confident 'no matching client data'."""
    install, _ = setup
    exact = "SELECT c.name FROM clients c JOIN holdings h ON c.client_id = h.client_id JOIN instruments i ON h.instrument_id = i.instrument_id WHERE i.name = 'Test World'"
    llm = install(Route=[R("portfolio")], SqlQuery=[multi.SqlQuery(sql=exact), multi.SqlQuery(sql=GOOD_SQL)],
                  FinalAnswer=[FA("10 clients.")], Verification=[V(True)])
    out = multi.run_multi("Which clients hold the Test World fund?")
    assert llm.count("SqlQuery") == 2 and out["sql"] == GOOD_SQL
    assert any("compares an instrument name with =" in step for step in out["trace"])
    retry_prompt = [p for s, p in llm.prompts if s == "SqlQuery"][1]
    assert "use lower(instruments.name) LIKE" in retry_prompt


def test_exact_name_guard_does_not_fire_for_unknown_clients_or_non_empty_results():
    assert multi._INSTRUMENT_NAME_EQ.search("... WHERE i2.name = 'Xtrackers MSCI World' ...")
    assert multi._INSTRUMENT_NAME_EQ.search("... WHERE instruments.name = 'X'")
    assert not multi._INSTRUMENT_NAME_EQ.search("... WHERE c.name = 'Nobody' ...")
    assert not multi._INSTRUMENT_NAME_EQ.search("... WHERE clients.name = 'Nobody' ...")


def test_sql_prompt_asks_for_like_on_instrument_names():
    assert "never compare an instrument name with =" in multi.SQL_PROMPT
