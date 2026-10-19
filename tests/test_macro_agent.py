"""Control-flow tests for the macro agent. The LLM and the retriever are scripted fakes: no API, no models."""
import pytest

from wealthai.agent import macro
from wealthai.agent.guardrails import ABSTAIN_MSG, REFUSE_MSG

PASSAGE = {"text": "[SNB Monetary policy assessment of 24 September 2026 | PDF p.1] The SNB left the policy rate at 0%.",
           "source": "snb_monetary_policy_assessment_2026_09_24.pdf", "section": "SNB Monetary policy assessment of 24 September 2026"}


class FakeLLM:
    def __init__(self, **by_schema):
        self.queues = {name: list(items) for name, items in by_schema.items()}
        self.calls = []
        self.prompts = []

    def __call__(self, prompt, schema=None, **kwargs):
        self.calls.append(schema.__name__)
        self.prompts.append((schema.__name__, prompt))
        queue = self.queues[schema.__name__]
        return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture
def setup(monkeypatch):
    retrievals = []

    def fake_retrieve(query, k=5, mode="hybrid_rerank"):
        retrievals.append((query, k, mode))
        return [dict(PASSAGE)]

    monkeypatch.setattr(macro, "retrieve_macro", fake_retrieve)
    monkeypatch.setattr(macro, "_graph", None)

    def install(**scripts):
        llm = FakeLLM(**scripts)
        monkeypatch.setattr(macro, "generate", llm)
        return llm

    return install, retrievals


def Q(text="SNB policy rate"):
    return macro.SearchQuery(query=text)


def G(relevant, missing=""):
    return macro.Grade(relevant=relevant, missing=missing)


def A(text="The SNB left the policy rate at 0% (24 September 2026).", answerable=True):
    return macro.MacroAnswer(answerable=answerable, answer=text, citations=[1] if answerable else [])


def V(supported, unsupported=""):
    return macro.Verification(supported=supported, unsupported=unsupported)


def test_happy_path_runs_every_node_once(setup):
    install, retrievals = setup
    llm = install(SearchQuery=[Q()], Grade=[G(True)], MacroAnswer=[A()], Verification=[V(True)])
    out = macro.run_macro_agent("What did the SNB decide in September 2026?")
    assert llm.calls == ["SearchQuery", "Grade", "MacroAnswer", "Verification"]
    assert out["answerable"] and out["citations"] == [1] and out["mode"] == "macro"
    assert out["contexts"] == [PASSAGE["text"]] and out["sources"] == [PASSAGE["source"]]
    assert retrievals == [("SNB policy rate", 5, "hybrid")]


def test_advice_is_refused_without_any_llm_call(setup):
    install, retrievals = setup
    llm = install(SearchQuery=[Q()])
    out = macro.run_macro_agent("Should I buy Swiss bonds after the SNB decision?")
    assert out["answer"] == REFUSE_MSG and not out["answerable"]
    assert llm.calls == [] and retrievals == []


def test_irrelevant_passages_trigger_one_rewrite_with_feedback_then_success(setup):
    install, retrievals = setup
    llm = install(SearchQuery=[Q("first"), Q("second")], Grade=[G(False, "August value"), G(True)],
                  MacroAnswer=[A()], Verification=[V(True)])
    out = macro.run_macro_agent("What was CPI inflation in August 2026?")
    assert [r[0] for r in retrievals] == ["first", "second"]
    assert retrievals[1][1] > retrievals[0][1]  # the net is widened on the retry
    assert "August value" in llm.prompts[2][1]  # the second rewrite sees what was missing
    assert out["answerable"]


def test_abstains_after_the_retrieval_budget_is_spent(setup):
    install, retrievals = setup
    llm = install(SearchQuery=[Q()], Grade=[G(False, "nothing")], MacroAnswer=[A()], Verification=[V(True)])
    out = macro.run_macro_agent("What is the level of the Dow Jones today?")
    assert len(retrievals) == macro.MAX_RETRIEVALS
    assert out["answer"] == ABSTAIN_MSG and not out["answerable"] and out["citations"] == []
    assert "MacroAnswer" not in llm.calls


def test_unsupported_answer_is_regenerated_once_with_the_complaint(setup):
    install, _ = setup
    llm = install(SearchQuery=[Q()], Grade=[G(True)], MacroAnswer=[A("bad"), A("good")],
                  Verification=[V(False, "number 3% not in passages"), V(True)])
    out = macro.run_macro_agent("What is the policy rate?")
    assert out["answer"] == "good"
    second = [p for s, p in llm.prompts if s == "MacroAnswer"][1]
    assert "number 3% not in passages" in second


def test_still_unsupported_after_the_regeneration_abstains(setup):
    install, _ = setup
    llm = install(SearchQuery=[Q()], Grade=[G(True)], MacroAnswer=[A("bad")], Verification=[V(False, "x")])
    out = macro.run_macro_agent("What is the policy rate?")
    assert out["answer"] == ABSTAIN_MSG and not out["answerable"]
    assert llm.calls.count("MacroAnswer") == 2


def test_an_honest_not_answerable_is_not_verified(setup):
    install, _ = setup
    llm = install(SearchQuery=[Q()], Grade=[G(True)],
                  MacroAnswer=[A("The documents do not contain this.", answerable=False)], Verification=[V(True)])
    out = macro.run_macro_agent("What will inflation be in 2035?")
    assert not out["answerable"] and "Verification" not in llm.calls
