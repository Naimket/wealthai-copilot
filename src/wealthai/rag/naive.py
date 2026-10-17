"""Naive RAG baseline: dense top-k -> prompt -> grounded answer with citations."""
from pydantic import BaseModel, Field

from wealthai.ingestion.index import search
from wealthai.llm import generate

PROMPT = """You are an assistant for wealth advisors. Answer the question using ONLY the numbered
context passages below (excerpts of fund Key Information Documents).

Rules:
- If the passages do not contain the answer, set answerable=false and say the documents do not
  contain it. Never use outside knowledge and never guess numbers.
- If the question asks for personal investment advice or a recommendation to buy/sell, set
  answerable=false and explain you can only report facts from the documents.
- Name the fund each figure belongs to. Be concise.
- List in `citations` the numbers of the passages you used.

Context:
{context}

Question: {question}"""


class RagAnswer(BaseModel):
    answerable: bool = Field(description="True only if the context supports an answer")
    answer: str
    citations: list[int] = Field(description="Numbers of the context passages used")


def answer_question(question: str, k: int = 5) -> dict:
    hits = search(question, k=k)
    contexts = [h.payload["text"] for h in hits]
    context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    result = generate(PROMPT.format(context=context, question=question), schema=RagAnswer)
    return {
        "question": question,
        "answer": result.answer,
        "answerable": result.answerable,
        "citations": result.citations,
        "contexts": contexts,
        "sources": [h.payload["source"] for h in hits],
        "sections": [h.payload["section"] for h in hits],
    }
