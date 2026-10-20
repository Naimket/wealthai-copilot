"""Multi-source advisor agent: routes a question to the KID agent, the macro agent, the client database, or both.

        guard --advice--> refuse
          |
        route --advice-------> refuse
          |   --out_of_scope-> out_of_scope
          |   --kid----------> kid_agent ---------------------------> END   (the evaluated KID agent, unchanged)
          |   --macro--------> macro_agent --------------------------> END   (SNB publications, own retrieval)
          |   --kid_first----> kid_first_question --> kid_agent --> select_instruments --> sql_gen
          |                    (KID facts choose the instruments, validated against the database; then SQL finds the clients)
          |
        sql_gen <--(error, once)-- sql_run <--+
          |                          |
          +--------------------------+  route=portfolio --------------+
                                     |                                |
                                     +-- route=both --> kid_question -+-- needed --> kid_agent
                                                                      |                  |
                                                  compose <-----------+------------------+
                                                     |
                                                  verify_final --supported--> END
                                                     ^   |
                                                     +---+-- once; still unsupported --> abstain

The SQL is written by the LLM but runs through tools.sql (read-only, whitelisted). The final answer of a
multi-source question is checked against ALL the evidence (database rows + KID answer + KID passages).
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from wealthai.agent.guardrails import ABSTAIN_MSG, REFUSE_MSG, is_advice_request
from wealthai.llm import generate
from wealthai.tools.sql import DEFAULT_DB, SCHEMA_DESCRIPTION, SqlToolError, run_query

DB_PATH = DEFAULT_DB
MAX_SQL_ATTEMPTS = 2  # first query + 1 retry with the error message
OUT_OF_SCOPE_MSG = ("I can answer questions about the fund KIDs, the SNB publications of September 2026 and the "
                    "client portfolios in this system. "
                    "This question is outside that scope.")
SQL_FAILED_MSG = "I could not retrieve the portfolio data needed to answer this question."
ROUTES = {"kid", "macro", "portfolio", "both", "kid_first", "advice", "out_of_scope"}

ROUTE_PROMPT = """You route questions from a wealth advisor to the right knowledge source.

Sources:
- KIDs: Key Information Documents of five ETFs (iShares Core MSCI EM IMI, iShares Core MSCI World,
  iShares Core S&P 500, Vanguard FTSE All-World, Xtrackers MSCI World): costs, risk indicator (SRI),
  recommended holding period, performance scenarios, what happens if the provider fails.
- SNB: two Swiss National Bank publications of September 2026 (monetary policy assessment of 24 September 2026 and
  Quarterly Bulletin 3/2026): policy rate, Swiss inflation, GDP, labour market, exchange rates, interest rates,
  mortgage reference rate and rents, inflation expectations (surveys of households, companies and analysts, for
  example Consensus Economics), residential real estate prices, the Swiss stock market (SMI, sector indices),
  money and credit aggregates (monetary base, M1-M3, mortgage lending), SNB inflation forecast, the global economic
  environment (United States, euro area, China: growth, unemployment, inflation, tariffs, oil prices, Fed and ECB
  rates), chronicle of monetary events, glossary of terms.
- Portfolio database: fictional clients (age, city, risk profile, horizon) and their holdings and values.

Choose exactly one route:
- kid: needs only facts from the KIDs; no client is involved.
- macro: needs only facts from the SNB publications (monetary policy, Swiss or foreign economy, inflation, rates, franc, tariffs); no fund or client is involved.
- portfolio: needs only client data (who holds what, values, allocations, rankings or counts of clients).
- both: needs client data AND KID facts about funds the question already points to through a client (for example the
  costs or risk of the funds a client holds).
- kid_first: KID facts decide WHICH funds qualify (the cheapest, the lowest risk, costs above a threshold) and the
  question then asks which clients hold them.
- advice: asks what to buy, sell or choose, or whether something is suitable or recommended for a client.
- out_of_scope: anything else (market news, live prices, general chat). A question about a macro figure that the
  SNB publications could contain is macro, even if the answer may turn out not to be there.

Question: {question}"""

SQL_PROMPT = """Write ONE SQLite SELECT query that answers the question.

{schema}
Rules:
- Match clients by name with: lower(clients.name) LIKE lower('%name as written%').
- Portfolio value = SUM(holdings.market_value_chf). Round CHF amounts to 0 decimals and averages and percentages to 1 decimal; never round counts or ages. Use shares of a total when asked for allocations.
- Return only the columns needed, with readable aliases. End with LIMIT 50 unless you aggregate to one row.
- Use LIMIT 1 or another small LIMIT only when the question asks for a top-N or a single extreme (largest, oldest, ...).
- The database has NO cost, fee or risk columns: those facts exist only in the KIDs. Never invent a column. If the question needs KID facts to choose instruments, return the clients and the instruments they hold (name, isin, kid_source).
- If the question also needs KID facts about the instruments, include instruments.name, instruments.isin and instruments.kid_source in the result.
- Do not answer the question; only write the query.
{kid_facts}{feedback}
Question: {question}"""

KID_FIRST_PROMPT = """The advisor asked the question below. Which funds qualify depends on facts printed in the fund documents (KIDs).
Write ONE self-contained question for the KID assistant that asks for that fact for EACH of the five funds
(iShares Core MSCI EM IMI, iShares Core MSCI World, iShares Core S&P 500, Vanguard FTSE All-World, Xtrackers MSCI World),
for example "What are the ongoing costs of each of the five funds?". Ask only for the fact the selection depends on.

Advisor question: {question}"""

SELECT_PROMPT = """The advisor asked the question below. The KID answer lists a fact for each fund.
Choose the instruments that satisfy the condition in the question (for example the lowest cost, or a cost above a
threshold), comparing the numbers yourself. Return their names EXACTLY as written in the list of instruments.
If the KID answer does not give the fact for the funds, return an empty list.

Advisor question: {question}

KID ANSWER:\n{kid}

INSTRUMENTS WITH A KID (name | isin):
{instruments}"""

KID_QUESTION_PROMPT = """The advisor asked the question below. The database result shows the client's data.
Write ONE self-contained question for the fund-document (KID) assistant about the instruments in the result.
- Use the exact instrument names from the result, only for instruments whose kid_source is not empty.
- Ask only for the KID facts the advisor needs (for example ongoing costs, risk indicator, holding period).
- If no instrument in the result has a KID, set needed=false.

Advisor question: {question}

Database result:
{table}"""

COMPOSE_PROMPT = """You are an assistant for wealth advisors. Answer the question using ONLY the evidence below.

DATABASE RESULT (fictional client data, amounts in CHF):
{table}

SQL THAT PRODUCED IT (its WHERE, GROUP BY and LIMIT define what each number or row refers to):
{sql}

{qualifying}KID ANSWER (from the fund documents):
{kid}

Rules:
- When FUNDS THAT QUALIFY is given, the condition in the question (the lowest cost, costs above a threshold, ...) is
  established by the KID ANSWER; do not refuse it for lack of cost data in the database. Report the clients in the database result.
- Never use outside knowledge and never guess a number. Do the arithmetic only from the figures shown.
- If the database result has no rows, say no matching client data was found.
- For an instrument whose kid_source is empty, say that no KID is available in the knowledge base for it.
- If the KID answer says the documents lack something, say so instead of filling the gap.
- Do not give investment advice or recommendations. Be concise and name the client and fund each figure belongs to.
- Write amounts as CHF with thousands separators (for example CHF 1,494,694).
- Describe each figure by what the query selected: a count produced with WHERE city = 'Geneva' is the number of clients in Geneva, even if the result table shows only the number.
- If the question asks for a ranking or an extreme (lowest, highest, cheapest, most, best) and the evidence does not directly establish it, say that the available data does not allow that conclusion. Never answer about a different subset than the one asked (for example a single row picked by LIMIT 1).
- If the question asks which clients or funds, name every one in the database result; never shorten the list.
- Set answerable=false only if the evidence cannot answer the question.
{feedback}
Question: {question}"""

VERIFY_PROMPT = """Check the ANSWER against the EVIDENCE. supported=true only if every number, name and claim in the answer
appears in, or follows directly from, the evidence (re-do any sum, share or ranking yourself). The SQL shows what
each number or row refers to. Also set supported=false if the answer addresses a different subset or question than
the one asked (for example one LIMIT 1 row presented as the answer to a ranking question), and if the question asks
which clients or funds and the answer leaves out any in the DATABASE RESULT or names one that is not there.
When FUNDS THAT QUALIFY is given, the choice of funds counts as supported if the figures in the KID ANSWER justify it
(re-do the comparison yourself); the clients must be those in the DATABASE RESULT.
List any unsupported claim in `unsupported`.

Question: {question}

DATABASE RESULT (fictional client data; all amounts are in CHF):
{table}

SQL THAT PRODUCED IT:
{sql}

{qualifying}KID ANSWER:
{kid}

KID PASSAGES:
{passages}

ANSWER: {answer}"""


class Route(BaseModel):
    route: str = Field(description="One of: kid, macro, portfolio, both, advice, out_of_scope")
    reason: str = Field(default="", description="One short sentence")


class InstrumentSelection(BaseModel):
    names: list[str] = Field(default_factory=list, description="Instrument names, exactly as listed")
    reason: str = Field(default="", description="One short sentence")


class SqlQuery(BaseModel):
    sql: str = Field(description="A single SQLite SELECT statement")


class KidQuestion(BaseModel):
    needed: bool
    question: str = Field(default="", description="Self-contained question for the KID assistant")


class FinalAnswer(BaseModel):
    answerable: bool
    answer: str


class Verification(BaseModel):
    supported: bool
    unsupported: str = Field(default="", description="Unsupported claims, if any")


class State(TypedDict, total=False):
    question: str
    route: str
    sql: str
    sql_attempts: int
    sql_error: str
    table: str
    kid_question: str
    kid: dict
    macro: dict
    qualifying: list[dict]
    answer: str
    answerable: bool
    drafts: int
    unsupported: str
    supported: bool
    trace: Annotated[list[str], operator.add]


def run_kid_agent(question: str) -> dict:
    """The KID agent (imported lazily: loading it also loads the embedding and reranking models)."""
    from wealthai.agent.graph import run_agent

    return run_agent(question)


def _kid_text(state: State) -> str:
    kid = state.get("kid")
    if not kid:
        return "(not requested)"
    return kid["answer"] if kid["answerable"] else f"The KID assistant could not answer: {kid['answer']}"


def _qualifying_text(state: State) -> str:
    """The funds chosen from the KID facts (kid_first route only); empty on every other route."""
    chosen = state.get("qualifying")
    if not chosen:
        return ""
    return ("FUNDS THAT QUALIFY (chosen from the KID ANSWER by comparing its figures; the database result lists the clients\n"
            "holding them): " + "; ".join(c["name"] for c in chosen) + "\n\n")


def _passages(state: State) -> str:
    kid = state.get("kid")
    if not kid or not kid["contexts"]:
        return "(none)"
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(kid["contexts"], start=1))


# ---- nodes -------------------------------------------------------------------------------------------------------
def guard(state: State) -> dict:
    return {"sql_attempts": 0, "drafts": 0, "trace": ["guard"]}


def refuse(state: State) -> dict:
    return {"route": "advice", "answer": REFUSE_MSG, "answerable": False, "trace": ["refuse (advice request)"]}


def route(state: State) -> dict:
    out = generate(ROUTE_PROMPT.format(question=state["question"]), schema=Route)
    chosen = out.route.strip().lower()
    if chosen not in ROUTES:
        chosen = "kid"  # the KID agent abstains by itself when the documents cannot answer
    return {"route": chosen, "trace": [f"route -> {chosen} ({out.reason})"]}


def out_of_scope(state: State) -> dict:
    return {"answer": OUT_OF_SCOPE_MSG, "answerable": False, "trace": ["out_of_scope"]}


def kid_agent(state: State) -> dict:
    question = state.get("kid_question") or state["question"]
    kid = run_kid_agent(question)
    trace = [f"kid_agent({question!r})"] + [f"    {step}" for step in kid["trace"]]
    update = {"kid": kid, "trace": trace}
    if state["route"] == "kid":  # nothing to combine: the KID answer is the answer
        update.update(answer=kid["answer"], answerable=kid["answerable"])
    return update


def run_macro_agent(question: str) -> dict:
    """The macro agent (imported lazily: loading it also loads the embedding model)."""
    from wealthai.agent.macro import run_macro_agent as _run

    return _run(question)


def macro_agent(state: State) -> dict:
    macro = run_macro_agent(state["question"])
    trace = [f"macro_agent({state['question']!r})"] + [f"    {step}" for step in macro["trace"]]
    return {"macro": macro, "answer": macro["answer"], "answerable": macro["answerable"], "trace": trace}


def sql_gen(state: State) -> dict:
    feedback = f"Your previous query failed: {state['sql_error']}. Fix it." if state.get("sql_error") else ""
    kid_facts = ""
    if state.get("route") == "kid_first" and state.get("kid"):
        isins = ", ".join(f"'{q['isin']}'" for q in state.get("qualifying", []))
        names = "; ".join(q["name"] for q in state.get("qualifying", []))
        kid_facts = ("KID FACTS (from the fund documents): " + _kid_text(state) + "\n"
                     f"The funds that qualify for the question, already decided from these facts: {names}.\n"
                     f"Select the DISTINCT clients holding them and filter with instruments.isin IN ({isins}); do not filter by name\n"
                     "with LIKE and do not apply any other cost or risk condition. Return client names only, one row per client.\n")
    prompt = SQL_PROMPT.format(schema=SCHEMA_DESCRIPTION, question=state["question"], kid_facts=kid_facts, feedback=feedback)
    out = generate(prompt, schema=SqlQuery)
    return {"sql": out.sql.strip(), "sql_attempts": state.get("sql_attempts", 0) + 1, "trace": [f"sql_gen -> {out.sql.strip()!r}"]}


def sql_run(state: State) -> dict:
    try:
        result = run_query(state["sql"], db_path=DB_PATH)
    except SqlToolError as exc:
        return {"sql_error": str(exc), "trace": [f"sql_run error: {exc}"]}
    return {"sql_error": "", "table": result.to_markdown(), "trace": [f"sql_run ok -> {len(result.rows)} rows"]}


def sql_failed(state: State) -> dict:
    return {"answer": SQL_FAILED_MSG, "answerable": False, "trace": ["sql_failed"]}


def kid_first_question(state: State) -> dict:
    out = generate(KID_FIRST_PROMPT.format(question=state["question"]), schema=KidQuestion)
    question = out.question.strip() or state["question"]  # fall back to the advisor question itself
    return {"kid_question": question, "trace": [f"kid_first_question -> {question!r}"]}


def select_instruments(state: State) -> dict:
    """The KID facts decide which instruments qualify; the names are validated against the database in code."""
    known = run_query("SELECT name, isin FROM instruments WHERE kid_source IS NOT NULL ORDER BY name", db_path=DB_PATH).rows
    listing = "\n".join(f"{name} | {isin}" for name, isin in known)
    out = generate(SELECT_PROMPT.format(question=state["question"], kid=_kid_text(state), instruments=listing),
                   schema=InstrumentSelection)
    by_name = {name.strip().lower(): {"name": name, "isin": isin} for name, isin in known}
    chosen = []
    for name in out.names:
        hit = by_name.get(name.strip().lower())
        if hit and hit not in chosen:
            chosen.append(hit)
    return {"qualifying": chosen, "trace": [f"select_instruments -> {[c['name'] for c in chosen]} ({out.reason})"]}


def kid_question(state: State) -> dict:
    out = generate(KID_QUESTION_PROMPT.format(question=state["question"], table=state["table"]), schema=KidQuestion)
    if not out.needed or not out.question.strip():
        return {"kid_question": "", "trace": ["kid_question -> not needed (no KID for these instruments)"]}
    return {"kid_question": out.question.strip(), "trace": [f"kid_question -> {out.question.strip()!r}"]}


def compose(state: State) -> dict:
    feedback = ""
    if state.get("unsupported"):
        feedback = f"Your previous draft had unsupported claims: {state['unsupported']}. Remove them; use only the evidence."
    prompt = COMPOSE_PROMPT.format(table=state["table"], sql=state.get("sql", ""), kid=_kid_text(state),
                                   qualifying=_qualifying_text(state),
                                   question=state["question"], feedback=feedback)
    out = generate(prompt, schema=FinalAnswer)
    drafts = state.get("drafts", 0) + 1
    return {"answer": out.answer, "answerable": out.answerable, "drafts": drafts,
            "trace": [f"compose (draft {drafts}) answerable={out.answerable}"]}


def verify_final(state: State) -> dict:
    prompt = VERIFY_PROMPT.format(question=state["question"], table=state["table"], sql=state.get("sql", ""),
                                  kid=_kid_text(state), qualifying=_qualifying_text(state), passages=_passages(state),
                                  answer=state["answer"])
    out = generate(prompt, schema=Verification)
    return {"supported": out.supported, "unsupported": out.unsupported,
            "trace": [f"verify_final supported={out.supported} {out.unsupported!r}"]}


def abstain(state: State) -> dict:
    return {"answer": ABSTAIN_MSG, "answerable": False, "trace": ["abstain"]}


# ---- routing -----------------------------------------------------------------------------------------------------
def after_guard(state: State) -> str:
    return "refuse" if is_advice_request(state["question"]) else "route"


def after_route(state: State) -> str:
    return {"kid": "kid_agent", "macro": "macro_agent", "portfolio": "sql_gen", "both": "sql_gen", "kid_first": "kid_first_question",
            "advice": "refuse", "out_of_scope": "out_of_scope"}[state["route"]]


def after_sql_run(state: State) -> str:
    if state.get("sql_error"):
        return "sql_gen" if state["sql_attempts"] < MAX_SQL_ATTEMPTS else "sql_failed"
    return "kid_question" if state["route"] == "both" else "compose"


def after_kid_question(state: State) -> str:
    return "kid_agent" if state.get("kid_question") else "compose"


def after_kid_agent(state: State) -> str:
    if state["route"] == "kid":
        return END
    if state["route"] == "kid_first":  # no KID facts, no way to choose the instruments: do not guess
        return "select_instruments" if state["kid"]["answerable"] else "abstain"
    return "compose"


def after_select(state: State) -> str:
    return "sql_gen" if state.get("qualifying") else "abstain"  # nothing qualifies or the facts are missing: do not guess


def after_compose(state: State) -> str:
    return "verify_final" if state["answerable"] else END


def after_verify(state: State) -> str:
    if state["supported"]:
        return END
    return "compose" if state["drafts"] < 2 else "abstain"  # one regeneration, then give up


def build_graph():
    g = StateGraph(State)
    for name, fn in [("guard", guard), ("refuse", refuse), ("route", route), ("out_of_scope", out_of_scope),
                     ("kid_agent", kid_agent), ("macro_agent", macro_agent), ("kid_first_question", kid_first_question), ("select_instruments", select_instruments),
                     ("sql_gen", sql_gen), ("sql_run", sql_run), ("sql_failed", sql_failed),
                     ("kid_question", kid_question), ("compose", compose), ("verify_final", verify_final),
                     ("abstain", abstain)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", after_guard, {"refuse": "refuse", "route": "route"})
    g.add_conditional_edges("route", after_route, {"kid_agent": "kid_agent", "macro_agent": "macro_agent",
                                                   "sql_gen": "sql_gen", "kid_first_question": "kid_first_question", "refuse": "refuse",
                                                   "out_of_scope": "out_of_scope"})
    g.add_edge("sql_gen", "sql_run")
    g.add_conditional_edges("sql_run", after_sql_run, {"sql_gen": "sql_gen", "sql_failed": "sql_failed",
                                                       "kid_question": "kid_question", "compose": "compose"})
    g.add_conditional_edges("kid_question", after_kid_question, {"kid_agent": "kid_agent", "compose": "compose"})
    g.add_edge("kid_first_question", "kid_agent")
    g.add_conditional_edges("kid_agent", after_kid_agent, {END: END, "compose": "compose", "select_instruments": "select_instruments",
                                                "abstain": "abstain"})
    g.add_conditional_edges("select_instruments", after_select, {"sql_gen": "sql_gen", "abstain": "abstain"})
    g.add_conditional_edges("compose", after_compose, {"verify_final": "verify_final", END: END})
    g.add_conditional_edges("verify_final", after_verify, {"compose": "compose", "abstain": "abstain", END: END})
    for terminal in ("refuse", "out_of_scope", "macro_agent", "sql_failed", "abstain"):
        g.add_edge(terminal, END)
    return g.compile()


_graph = None


def run_multi(question: str) -> dict:
    """Run the multi-source agent. Same keys as the KID agent, plus route, sql and the full trace.

    `contexts` holds the KID passages and, when the database was used, one extra entry with the SQL result,
    so the faithfulness of multi-source answers can be measured with the same tooling.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    s = _graph.invoke({"question": question})
    kid = s.get("kid") or s.get("macro") or {}
    contexts = list(kid.get("contexts", []))
    sources = list(kid.get("sources", []))
    sections = list(kid.get("sections", []))
    if s.get("table"):
        contexts.append(f"SQL: {s['sql']}\n{s['table']}")
        sources.append("portfolios.db")
        sections.append("sql result")
    return {
        "question": question,
        "answer": s["answer"],
        "answerable": s["answerable"],
        "citations": kid.get("citations", []) if s.get("route") in ("kid", "macro") else [],
        "contexts": contexts,
        "sources": sources,
        "sections": sections,
        "mode": "multi",
        "route": s.get("route", ""),
        "sql": s.get("sql", ""),
        "trace": s["trace"],
    }
