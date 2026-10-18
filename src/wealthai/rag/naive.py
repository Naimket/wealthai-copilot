"""RAG answer step: retrieve top-k (any mode, see retrieval.py) -> prompt -> grounded answer with citations."""
from pydantic import BaseModel, Field

from wealthai.llm import generate
from wealthai.retrieval import retrieve

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


def answer_question(question: str, k: int = 5, mode: str = "hybrid_filter_rerank") -> dict:
    hits = retrieve(question, k=k, mode=mode)
    contexts = [h["text"] for h in hits]
    context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    result = generate(PROMPT.format(context=context, question=question), schema=RagAnswer)
    return {
        "question": question,
        "answer": result.answer,
        "answerable": result.answerable,
        "citations": result.citations,
        "contexts": contexts,
        "sources": [h["source"] for h in hits],
        "sections": [h["section"] for h in hits],
        "mode": mode,
    }
