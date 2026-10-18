"""Streamlit front-end for WealthAI-Copilot. It only talks to the REST API (see src/wealthai/api.py).

Run (API first, in another terminal):
    uv run uvicorn wealthai.api:app --app-dir src
    uv run --group ui streamlit run app/streamlit_app.py
"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("WEALTHAI_API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 180  # the agent makes several sequential LLM calls
EXAMPLES = [
    "What are the ongoing costs of the Vanguard FTSE All-World UCITS ETF?",
    "Which of the iShares funds is cheapest?",
    "How many clients do we have per risk profile?",
    "Which funds does Alice Brunner hold and what are their ongoing costs?",
    "Should I sell Alice Brunner's Vanguard position?",
]
ROUTE_LABELS = {
    "kid": "Fund documents (KIDs)",
    "portfolio": "Client portfolios (SQL)",
    "both": "Portfolios + KIDs",
    "advice": "Refused: investment advice",
    "out_of_scope": "Out of scope",
}


def ask(question: str) -> dict:
    response = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="WealthAI-Copilot", layout="wide")
st.title("WealthAI-Copilot")
st.caption("Multi-source assistant for wealth advisors: fund KIDs and fictional client portfolios. "
           "It reports facts from its sources and does not give investment advice. All client data is synthetic.")

with st.sidebar:
    st.subheader("Examples")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["question"] = example
    st.divider()
    st.caption(f"API: {API_URL}")

question = st.text_area("Question", key="question", height=80, placeholder="Ask about a fund or a client...")

if st.button("Ask", type="primary", disabled=len(question.strip()) < 3):
    try:
        with st.spinner("Routing, retrieving and verifying..."):
            result = ask(question.strip())
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.json().get("detail", f"The API returned an error ({exc.response.status_code})."))
    except httpx.HTTPError:
        st.error(f"Could not reach the API at {API_URL}. Is it running?")
    else:
        st.markdown(f"**Source:** {ROUTE_LABELS.get(result['route'], result['route'] or 'n/a')}")
        (st.success if result["answerable"] else st.warning)(result["answer"])
        if result["sql"]:
            with st.expander("SQL executed (read-only)"):
                st.code(result["sql"], language="sql")
        if result["contexts"]:
            with st.expander(f"Evidence ({len(result['contexts'])} passages)"):
                for source, section, text in zip(result["sources"], result["sections"], result["contexts"]):
                    st.markdown(f"**{source}** | {section}")
                    st.text(text)
        with st.expander("Agent decision trace"):
            st.code("\n".join(result["trace"]))
