"""Deterministic compliance guardrails (no LLM). Shared by every agent graph."""
import re

ABSTAIN_MSG = "The provided documents do not contain enough information to answer this question reliably."
REFUSE_MSG = ("I can only report facts from the fund documents (KIDs). I cannot give personal investment "
              "advice or recommend buying or selling a product; that is a suitability decision for the advisor.")

# Deterministic compliance guardrail: runs before any LLM call.
ADVICE = re.compile(r"\bshould\b.{0,100}\b(buy|sell|invest|hold|choose|pick|switch)\b|\brecommend(?:ation)?s?\b", re.I | re.S)  # 'recommended holding period' must NOT match


def is_advice_request(question: str) -> bool:
    """True when the question asks for a recommendation or what to buy/sell/choose."""
    return ADVICE.search(question) is not None
