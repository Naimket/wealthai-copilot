"""Macro agent: answers factual questions from central-bank publications (SNB assessment and Quarterly Bulletin).

        guard --advice--> refuse
          |
        rewrite <---------------------+  (retry with feedback, max MAX_RETRIEVALS searches)
          |                           |
        retrieve --> grade --not enough--> (retry, else abstain)
                       |
                    generate --> verify --supported--> END
                                  |  (once; still unsupported --> abstain)

Same loop as the KID agent (agent/graph.py) with its own vocabulary, grading rules and retrieval (retrieval_macro.py):
macro questions are about a period, a country and a quantity, not about a fund. The KID agent is not touched.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from wealthai.agent.guardrails import ABSTAIN_MSG, REFUSE_MSG, is_advice_request
from wealthai.llm import generate
from wealthai.retrieval_macro import retrieve_macro

MAX_RETRIEVALS = 2
RETRIEVAL_MODE = "hybrid"  # on the macro set the reranker did not help (see docs/results.md)

REWRITE_PROMPT = """You turn a user question into a search query over Swiss National Bank (SNB) publications: the monetary
policy assessment of 24 September 2026 and Quarterly Bulletin 3/2026 (monetary policy report, business cycle signals,
chronicle of monetary events, glossary). Map the user's words to the vocabulary of these documents when relevant:
- key rate, interest rate, base rate -> "SNB policy rate"
- inflation, prices, CPI -> "CPI inflation" "consumer prices"
- growth, GDP -> "GDP" with the quarter or year
- unemployment, jobs -> "unemployment" "registered as unemployed" "labour market"
- franc, exchange rate, EUR/CHF -> "Swiss franc" "exchange rates"
- bond yields -> "Confederation bonds" "yield"
- inflation forecast -> "conditional inflation forecast"
- what does a term mean -> "Glossary" and the term
Rules: keep every country, date, month, quarter and number of the question; do not add facts or numbers; do not answer.
{feedback}
Question: {question}"""

GRADE_PROMPT = """Decide whether the passages contain what is needed to answer the question COMPLETELY.
- The passages must contain the SAME quantity for the SAME period, country and instrument that the question asks for
  (August is not May, the euro area is not the United States, the SNB policy rate is not SARON).
- The documents are SNB publications of September 2026. If the question asks for something they never contain
  (the level of another central bank's rate, live or intraday market data, index levels, events after
  24 September 2026, forecasts beyond the horizon they print, figures for individual companies), relevant=false.
- For comparisons, both figures must appear.
- If relevant=false, say in `missing` what is absent, in a few words.

Question: {question}

Passages:
{context}"""

ANSWER_PROMPT = """You are an assistant for wealth advisors. Answer the question using ONLY the numbered passages below
(excerpts of SNB publications from September 2026).

Rules:
- If the passages do not contain the answer, set answerable=false and say the documents do not contain it. Never use
  outside knowledge and never guess numbers.
- If the question asks for personal investment advice or a recommendation, set answerable=false and explain you can
  only report facts from the documents.
- State the period or date each figure refers to (for example "in August 2026") and, when it matters, whether it comes
  from the assessment or the bulletin. A conditional forecast depends on its stated assumption (policy rate held at 0%):
  say so when you report it.
- Do not extrapolate or predict beyond what the SNB states.
- Answer in the language of the question. Be concise.
- List in `citations` the numbers of the passages you used.

Context:
{context}

Question: {question}"""

VERIFY_PROMPT = """Check the ANSWER against the PASSAGES. supported=true only if every number, date, country and claim in the
answer appears in, or follows directly from, the passages (re-do any comparison yourself).
List any unsupported claim in `unsupported`.

Question: {question}

Passages:
{context}

ANSWER: {answer}"""


class SearchQuery(BaseModel):
    query: str = Field(description="Search query in the vocabulary of the SNB publications")


class Grade(BaseModel):
    relevant: bool
    missing: str = Field(default="", description="What is missing, if anything")


class MacroAnswer(BaseModel):
    answerable: bool = Field(description="True only if the passages support an answer")
    answer: str
    citations: list[int] = Field(description="Numbers of the passages used")


class Verification(BaseModel):
    supported: bool
    unsupported: str = Field(default="", description="Unsupported claims, if any")


class State(TypedDict, total=False):
    question: str
    search_query: str
    feedback: str
    attempt: int
    chunks: list[dict]
    relevant: bool
    answer: str
    answerable: bool
    citations: list[int]
    drafts: int
    unsupported: str
    supported: bool
    trace: Annotated[list[str], operator.add]


def _context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))


# ---- nodes -------------------------------------------------------------------------------------------------------
def guard(state: State) -> dict:
    return {"attempt": 0, "drafts": 0, "trace": ["guard"]}


def refuse(state: State) -> dict:
    return {"answer": REFUSE_MSG, "answerable": False, "citations": [], "chunks": [], "trace": ["refuse (advice request)"]}


def rewrite(state: State) -> dict:
    feedback = (f"A previous search did not find: {state['feedback']}. Use different wording."
                if state.get("feedback") else "")
    out = generate(REWRITE_PROMPT.format(question=state["question"], feedback=feedback), schema=SearchQuery)
    return {"search_query": out.query, "trace": [f"rewrite -> {out.query!r}"]}


def retrieve_node(state: State) -> dict:
    attempt = state.get("attempt", 0)
    k = 5 + 3 * attempt  # widen the net on a retry
    chunks = retrieve_macro(state["search_query"], k=k, mode=RETRIEVAL_MODE)
    return {"chunks": chunks, "attempt": attempt + 1, "trace": [f"retrieve k={k} -> {len(chunks)} chunks"]}


def grade(state: State) -> dict:
    out = generate(GRADE_PROMPT.format(question=state["question"], context=_context(state["chunks"])), schema=Grade)
    return {"relevant": out.relevant, "feedback": out.missing, "trace": [f"grade relevant={out.relevant} {out.missing!r}"]}


def generate_node(state: State) -> dict:
    prompt = ANSWER_PROMPT.format(context=_context(state["chunks"]), question=state["question"])
    if state.get("unsupported"):
        prompt += f"\n\nYour previous draft had unsupported claims: {state['unsupported']}. Remove them; use only the passages."
    out = generate(prompt, schema=MacroAnswer)
    drafts = state.get("drafts", 0) + 1
    return {"answer": out.answer, "answerable": out.answerable, "citations": out.citations, "drafts": drafts,
            "trace": [f"generate (draft {drafts}) answerable={out.answerable}"]}


def verify(state: State) -> dict:
    out = generate(VERIFY_PROMPT.format(question=state["question"], context=_context(state["chunks"]),
                                        answer=state["answer"]), schema=Verification)
    return {"supported": out.supported, "unsupported": out.unsupported,
            "trace": [f"verify supported={out.supported} {out.unsupported!r}"]}


def abstain(state: State) -> dict:
    return {"answer": ABSTAIN_MSG, "answerable": False, "citations": [], "trace": ["abstain"]}


# ---- routing -----------------------------------------------------------------------------------------------------
def after_guard(state: State) -> str:
    return "refuse" if is_advice_request(state["question"]) else "rewrite"


def after_grade(state: State) -> str:
    if state["relevant"]:
        return "generate"
    return "rewrite" if state["attempt"] < MAX_RETRIEVALS else "abstain"


def after_generate(state: State) -> str:
    return "verify" if state["answerable"] else END  # an honest "the documents do not say" needs no verification


def after_verify(state: State) -> str:
    if state["supported"]:
        return END
    return "generate" if state["drafts"] < 2 else "abstain"  # one regeneration, then give up


def build_graph():
    g = StateGraph(State)
    for name, fn in [("guard", guard), ("refuse", refuse), ("rewrite", rewrite), ("retrieve", retrieve_node),
                     ("grade", grade), ("generate", generate_node), ("verify", verify), ("abstain", abstain)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", after_guard, {"refuse": "refuse", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"})
    g.add_conditional_edges("generate", after_generate, {"verify": "verify", END: END})
    g.add_conditional_edges("verify", after_verify, {"generate": "generate", "abstain": "abstain", END: END})
    g.add_edge("refuse", END)
    g.add_edge("abstain", END)
    return g.compile()


_graph = None


def run_macro_agent(question: str) -> dict:
    """Same keys as the KID agent (answer, answerable, citations, contexts, sources, sections, trace)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    s = _graph.invoke({"question": question})
    chunks = s.get("chunks", [])
    return {
        "question": question,
        "answer": s["answer"],
        "answerable": s["answerable"],
        "citations": s.get("citations", []),
        "contexts": [c["text"] for c in chunks],
        "sources": [c["source"] for c in chunks],
        "sections": [c["section"] for c in chunks],
        "mode": "macro",
        "trace": s["trace"],
    }
