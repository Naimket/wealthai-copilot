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

## Phase 4 - retrieval ablation (35 single-document questions, k=5, no LLM calls)

| Mode | Factual hit@5 | Hard hit@5 | All hit@5 | MRR (all) |
|---|---|---|---|---|
| dense (baseline) | 100% | 66.7% | 85.7% | 0.764 |
| bm25 | 100% | 60.0% | 82.9% | 0.779 |
| hybrid (dense + BM25, RRF) | 100% | 80.0% | 91.4% | 0.818 |
| dense + rerank | 100% | 100% | 100% | 0.986 |
| hybrid + rerank | 100% | 100% | 100% | 0.986 |
| hybrid + fund filter + rerank | 100% | 100% | 100% | 0.986 |

The cross-encoder reranker is the component that moves the metric. The three reranked variants tie:
the set is saturated, so no claim is made about their relative order. Reranking costs ~1.3 s/query on CPU
(vs ~0.05 s for dense search).

## Phase 4 - end to end (55 questions, Gemini flash-lite)

| Retrieval | Comparative | Hard (answer) | Hard (retrieval) | Abstentions | Faithfulness | Answer relevancy |
|---|---|---|---|---|---|---|
| dense (Phase 3) | 75% | 58% (11/19) | 67% | 12/12 | 0.977 | 0.920 |
| hybrid + filter + rerank | 75% | 95% (18/19) | 100% | 12/12 | - | - |
| hybrid + per-fund + rerank | 100% | 100% (19/19) | 100% | 12/12 | 0.995 | 0.927 |

Per-fund balancing fixed a comparison question where the filter found both funds but the top-5 held
chunks of only one of them.

### Caveats
- These 55 questions were used to guide the design; 100% here is a development-set number, not a
  generalisation claim. A blind test set is planned for the final evaluation.
- One question (TER of Vanguard) flipped between runs with identical inputs: LLM run-to-run variance.
  Final numbers will be reported over several runs.
- Remaining known weakness: vocabulary gap in generation ("TER" is not a term the KIDs use). Planned fix:
  query rewriting in the LangGraph pipeline.
