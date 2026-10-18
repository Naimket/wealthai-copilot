"""REST API for the advisor copilot.

Run:  uv run uvicorn wealthai.api:app --app-dir src --reload
Docs: http://localhost:8000/docs  (interactive OpenAPI)

POST /ask runs the multi-source agent (KIDs + client database) and returns the answer together with the
evidence the answer was built from (route, SQL, passages, decision trace), so a caller can audit it.
"""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wealthai import llm
from wealthai.agent.multi import run_multi

log = logging.getLogger("wealthai.api")

app = FastAPI(
    title="WealthAI-Copilot",
    version="0.1.0",
    description="Multi-RAG assistant for wealth advisors: fund KIDs plus fictional client portfolios. "
                "Reports facts only; it does not give investment advice.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000,
                          examples=["Which funds does Alice Brunner hold and what are their ongoing costs?"])


class AskResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    route: str = Field(description="kid, portfolio, both, advice or out_of_scope")
    sql: str = Field(description="SQL executed against the portfolio database, if any")
    citations: list[int]
    sources: list[str]
    sections: list[str]
    contexts: list[str] = Field(description="Evidence passages and SQL result the answer was built from")
    trace: list[str] = Field(description="Decision trace of the agent graph")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": llm.MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so slow LLM calls do not block the server.
    try:
        return run_multi(request.question.strip())
    except Exception as exc:  # noqa: BLE001 - the details go to the log, never to the client
        log.exception("agent failed")
        if getattr(exc, "code", None) == 429:
            raise HTTPException(status_code=429, detail="The LLM provider's rate limit was reached; try again later.")
        raise HTTPException(status_code=503, detail="The agent could not complete the request.")
