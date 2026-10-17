# Results log

## Phase 3 - naive RAG baseline (dense top-5 + Gemini flash-lite, 55 questions)

| Question type | Answer correct | Retrieval hit | Faithfulness | Answer relevancy |
|---|---|---|---|---|
| Factual (20) | 100% | 100% | 1.00 | 0.97 |
| Comparative (4) | 75% | - | 1.00 | 0.90 |
| Hard (23, 19 answerable) | 58% | 67% | 0.94 | 0.85 |
| Unanswerable (8 + 4 near-miss traps) | 100% correct abstention | - | - | - |

Ragas scores only the 36 questions the system answered (abstentions have no claims to verify),
so it cannot see wrongful abstentions; the deterministic metrics cover that gap.

### Findings
- Faithfulness is high (0.98): the model rarely invents facts. The bottleneck is retrieval, not generation.
- Vocabulary gap: KIDs say "ongoing costs", users say "TER / expense ratio / SRI" -> 5 wrongful abstentions.
- Cross-document questions (ranking, "how many funds...") fail: top-5 covers only 1-2 funds. One answer was
  confidently wrong ("none of the funds...").
- Planned fixes: hybrid BM25 + reranker (phase 4), query rewriting and per-fund retrieval in LangGraph (5),
  structured facts table for numeric aggregation (6).
