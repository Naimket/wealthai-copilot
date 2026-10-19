"""LangGraph agent for KID question answering.

        guard --advice--> refuse
          |
        rewrite <-----------+  (retry with feedback, max MAX_RETRIEVALS searches)
          |                 |
        retrieve --> grade -+-- not enough evidence & retries left
                       |
                       +-- not enough evidence & no retries left --> abstain
                       |
                     generate --> verify --supported--> END
                       ^            |
                       +-- once ----+-- still unsupported --> abstain

Every LLM node returns a small Pydantic object (structured output), so the control flow is decided by
typed fields, never by parsing free text.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from wealthai.agent.guardrails import ABSTAIN_MSG, ADVICE, REFUSE_MSG  # noqa: F401
from wealthai.llm import generate
from wealthai.rag.naive import PROMPT, RagAnswer
from wealthai.retrieval import retrieve

MAX_RETRIEVALS = 2  # first search + 1 retry with a rewritten query
RETRIEVAL_MODE = "hybrid_perfund_rerank"
# Compliance constants live in guardrails.py (shared with the multi-source agent).

GLOSSARY = """

Terminology: KIDs do not use the abbreviations TER, OCF or the phrase "expense ratio". The closest KID figure is
"ongoing costs" (Management fees and other administrative or operating costs, % of the investment per year);
transaction costs are a separate line. If the question uses one of those terms, answer with the KID's ongoing-costs
figure and state explicitly that this is the term the KID uses. SRI means the summary risk indicator (X out of 7).
If the question's fund reference matches several funds in the passages, report the figure for each of them, clearly labelled."""

REWRITE_PROMPT = """You turn a user question into a search query over a library of fund Key Information Documents (KIDs).
KIDs use fixed vocabulary. Map the user's words to it when relevant:
- fees, TER, OCF, expense ratio -> "ongoing costs" "Management fees and other administrative or operating costs"
- SRI, risk score, risk level, how risky -> "summary risk indicator" "classified as" "out of 7"
- holding period -> "Recommended holding period"
- scenarios, returns -> "Performance Scenarios" (favourable, moderate, unfavourable, stress)
- what if the provider fails -> "unable to pay out"
- total costs, all costs, cost in USD over N years, cheapest once everything is included -> "Costs over time" "Total costs" "If you exit after 1 year" "If you exit after 5 years" "Annual cost impact" (this is NOT the ongoing-costs line)
- entry fee, exit fee -> "Entry costs" "Exit costs"
Rules: keep every fund name EXACTLY as the user wrote it; do not add facts or numbers; do not answer.
Keep the words that distinguish the question (for example "total", "stress", "Switzerland", "depositary"); never replace them with a more general term.
{feedback}
Question: {question}"""

GRADE_PROMPT = """Decide whether the passages contain what is needed to answer the question COMPLETELY.
- For comparisons, rankings, counts or "which funds" questions, EVERY fund involved must appear with its figure.
- The passages must contain the SAME kind of figure the question asks for: ongoing costs are not total costs,
  and transaction costs are not ongoing costs. A related figure does not make the passages relevant.
- If the question asks which fund or KID has a property stated in the text (for example which KID is addressed to
  a country, which fund has a given depositary or manufacturer), relevant=true as soon as a passage shows a fund
  with that property; the other funds do not need to appear. This does not apply to rankings of figures.
- If the question asks for something a KID never contains (live prices, news, macro data, assets under
  management, past performance of a different fund, a fund that is not in the passages), relevant=false.
- If the question refers to a fund loosely (e.g. "the MSCI World ETF") and the passages cover EVERY fund that
  matches that description, relevant=true: the answer will report each matching fund. Do not ask for clarification.
- If relevant=false, say in `missing` what is absent, in a few words.

Question: {question}

Passages:
{context}"""

VERIFY_PROMPT = """Check the ANSWER against the PASSAGES. supported=true only if every number, fund name and claim in the
answer appears in, or follows directly from, the passages (re-do any counting or ranking yourself).
List any unsupported claim in `unsupported`.

Question: {question}

Passages:
{context}

ANSWER: {answer}"""


class SearchQuery(BaseModel):
    query: str = Field(description="Search query in KID vocabulary, fund names unchanged")


class Grade(BaseModel):
    relevant: bool
    missing: str = Field(default="", description="What is missing, if anything")


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
    feedback = f"A previous search did not find: {state['feedback']}. Use different wording and name the KID section." if state.get("feedback") else ""
    out = generate(REWRITE_PROMPT.format(question=state["question"], feedback=feedback), schema=SearchQuery)
    return {"search_query": out.query, "trace": [f"rewrite -> {out.query!r}"]}


def retrieve_node(state: State) -> dict:
    attempt = state.get("attempt", 0)
    k = 5 + 3 * attempt  # widen the net on a retry
    chunks = retrieve(state["search_query"], k=k, mode=RETRIEVAL_MODE)
    return {"chunks": chunks, "attempt": attempt + 1, "trace": [f"retrieve k={k} -> {len(chunks)} chunks"]}


def grade(state: State) -> dict:
    out = generate(GRADE_PROMPT.format(question=state["question"], context=_context(state["chunks"])), schema=Grade)
    return {"relevant": out.relevant, "feedback": out.missing, "trace": [f"grade relevant={out.relevant} {out.missing!r}"]}


def generate_node(state: State) -> dict:
    prompt = PROMPT.format(context=_context(state["chunks"]), question=state["question"]) + GLOSSARY
    if state.get("unsupported"):
        prompt += f"\n\nYour previous draft had unsupported claims: {state['unsupported']}. Remove them; use only the passages."
    out = generate(prompt, schema=RagAnswer)
    return {"answer": out.answer, "answerable": out.answerable, "citations": out.citations,
            "drafts": state.get("drafts", 0) + 1, "trace": [f"generate (draft {state.get('drafts', 0) + 1}) answerable={out.answerable}"]}


def verify(state: State) -> dict:
    out = generate(VERIFY_PROMPT.format(question=state["question"], context=_context(state["chunks"]), answer=state["answer"]), schema=Verification)
    return {"supported": out.supported, "unsupported": out.unsupported,
            "trace": [f"verify supported={out.supported} {out.unsupported!r}"]}


def abstain(state: State) -> dict:
    return {"answer": ABSTAIN_MSG, "answerable": False, "citations": [], "trace": ["abstain"]}


# ---- routing -----------------------------------------------------------------------------------------------------
def after_guard(state: State) -> str:
    return "refuse" if ADVICE.search(state["question"]) else "rewrite"


def after_grade(state: State) -> str:
    if state["relevant"]:
        return "generate"
    return "rewrite" if state["attempt"] < MAX_RETRIEVALS else "abstain"


def after_generate(state: State) -> str:
    return "verify" if state["answerable"] else END


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


def run_agent(question: str) -> dict:
    """Run the agent; returns the same keys as rag.naive.answer_question plus the decision trace."""
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
        "mode": "agent",
        "search_query": s.get("search_query", ""),
        "trace": s["trace"],
    }
