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

## Phase 5 - LangGraph agent (55 dev questions, Gemini flash-lite)

Graph: guard (advice refusal, rule-based) -> rewrite -> retrieve (hybrid + per-fund + rerank) -> grade
(retry once with feedback, else abstain) -> generate -> verify (regenerate once, else abstain).

| System | Factual | Comparative | Hard | Unanswerable / traps | LLM calls per question | Latency per question |
|---|---|---|---|---|---|---|
| Single-pass, dense (Phase 3) | 100% | 75% | 58% | 12/12 | 1 | ~2 s |
| Single-pass, hybrid + per-fund + rerank | 100% | 100% | 100% | 12/12 | 1 | ~2 s |
| Agent (first run) | 100% | 100% | 95% (18/19) | 12/12 | ~4 | ~17 s |
| Agent (after fixing the grading prompt for loosely named funds) | 100% | 100% | 100% | 12/12 | ~4 | ~17 s |

Agent behaviour over the 55 questions: 1 refused as advice request (0 LLM calls), 11 retried retrieval and
abstained (all questions with no answer in the KIDs, no false abstentions after the fix), 0 regenerations.

**Ragas for the agent** (answered questions of the 55 dev set; judge: Gemini flash-lite default, local embeddings):

| Question type | Faithfulness | Answer relevancy |
|---|---|---|
| Factual | 1.000 | 0.977 |
| Comparative | 1.000 | 0.913 |
| Hard | 0.904 | 0.911 |
| Overall | 0.957 | 0.942 |

0 NaN. For comparison, the single-pass hybrid + per-fund + rerank pipeline scored 0.995 / 0.927 overall.
The agent's lower faithfulness sits in the hard questions. The cause has not been investigated yet (per-question
scores are in `data/eval/ragas_agent.csv`); Ragas judges with an LLM, so on a small set a few low scores move the
mean. Treat the gap as a lead to examine, not as a finding.

### Findings
- On this dev set the agent ties the single-pass pipeline with ~4x the LLM calls and ~8x the latency.
  The grade node works as an "is this in the documents?" detector (11/11 correct abstentions).
- The verify node never fired. Not evidence that it works: it needs a test with deliberately wrong answers.
- The TER case ("TER" is not a KID term) was fixed by query rewriting plus a terminology glossary in the
  generator. That glossary can also be given to the single-pass pipeline; the fair comparison belongs in the
  blind test set.
- A grader prompt that was too strict made the agent abstain on an ambiguous question ("the MSCI World ETF")
  although both matching funds were retrieved. Fixed in the prompt; found only because the agent's
  decision trace is logged.

## Phase 6 - multi-source router (40 questions, Gemini 3.1 Flash Lite)

Set: `data/golden/router_set.jsonl`. Ground truth for client-data questions is computed by an oracle SQL query
against the same database, not typed by hand; KID facts use values read from the KIDs. No LLM judge. A trivial
agent that always answers "There are 25 clients." passes 2 of 40: two safety items whose only requirement is that
nothing is destroyed or leaked.

| Category | n | Route correct | Answer correct, first run | Answer correct, after fixes |
|---|---|---|---|---|
| KID only | 6 | 6/6 | 6/6 | 6/6 |
| Client data only | 14 | 14/14 | 10/14 | 14/14 |
| Client data + KIDs | 5 | 5/5 | 5/5 | 5/5 |
| Multi-hop (KID facts needed before the SQL) | 2 | 2/2 | 0/2 | 1/2 |
| Advice (must refuse) | 4 | 4/4 | 4/4 | 4/4 |
| Out of scope (must decline) | 4 | 4/4 | 4/4 | 4/4 |
| Safety (injection, secrets) | 3 | 3/3 | 3/3 | 3/3 |
| Client that does not exist | 2 | 2/2 | 2/2 | 2/2 |
| **Total** | **40** | **40/40** | **34/40** | **39/40** |

Mean latency 16-17 s per question (up to ~40 s when both sources are used). The database was intact after every
run (25 clients, 138 holdings), including the prompt-injection items.

### What the first run found (and what was changed)
- **Three failures, one cause.** The SQL was correct, but the answer-writing step only saw the result table, not the
  query. A bare `1` did not say "clients in Geneva", so the model replied that the information was missing. Fix: the
  composer and the verifier now receive the executed SQL.
- **Rounding.** The average age came back as 44.0 instead of 44.2: a rule meant for money was applied to ages.
  Fix: the SQL prompt now separates CHF amounts, averages/percentages and counts/ages.
- **A confident wrong answer.** For "which clients hold the fund with the lowest ongoing costs?" the query picked one
  arbitrary row (`LIMIT 1`) and the answer described a different fund. This is the failure that matters in a
  regulated domain. Fix: the composer must say when the evidence does not establish a ranking, and the verifier
  rejects answers about a different subset than the one asked. After the fix the system says the data does not allow
  that conclusion (counted as a failure by the check, because it expects the list of clients; it is a safe refusal,
  not a wrong answer).
- **An invented column** (`ongoing_costs_percent`) made one query fail twice and the system answered that it could
  not retrieve the data: a safe failure. After the prompt changes (which include a rule that the database has no
  cost columns) the same question passed; I did not isolate which change made the difference.

### Caveats
- The 40 questions were used to find and fix these problems, so the second column is a development-set result. The
  first-run column is the cleaner estimate; a blind set is still needed.
- Small n: one question is 2.5 points. Single run, no variance estimate.
- Number checks ask whether the value appears in the answer, not which entity it is attached to; small counts can
  match by chance.
- The graph always queries the database first. Questions that need KID facts to choose instruments (cheapest fund,
  costs above a threshold) are answered only if the instruments can be listed from the database; a KID-first route
  is not built.
- Measured with Gemini 3.1 Flash Lite because the 3.5 Flash Lite daily quota was exhausted; the KID tables above were
  measured with Gemini flash-lite (3.5). Do not compare the two sets of numbers across models.

## Phase 7 - blind test set (52 questions, agent, `gemini-flash-lite-latest`)

The set (`data/golden/blind_set.jsonl`) was written from the five KIDs after the development work and committed
before any system was run on it. Topics deliberately differ from the dev set: scenario-table amounts in USD,
total costs over 1 and 5 years, entry/exit/performance fees, objectives and replication, depositary, manufacturer,
ISIN, Swiss representative and paying agent, complaints, two questions in French, and 10 near-miss traps.
Every answer pattern was checked against the KID text.

**Run 1 (first and only run before any fix):**

| Question type | Retrieval hit | Answer correct |
|---|---|---|
| Factual (33) | 33/33 | 33/33 |
| Hard (5) | 5/5 | 5/5 |
| Comparative (4) | - | 1/4 |
| Unanswerable (10) | - | 10/10 correct abstention |
| Total | | 49/52 (94%) |

Agent behaviour: 1 refused as advice, 13 retried retrieval, 10 abstained (all 10 traps), 0 regenerations after verify.

### The three misses (all comparative)
- **b39, b40** ("lowest / highest total costs over 5 years"): the rewriter turned the question into "ongoing costs
  Management fees and other administrative or operating costs", so the "Costs over time" table never reached the
  generator. In b40 the grader accepted the passages (ongoing costs are a related figure) and the generator then said
  the 5-year total was not in the documents; in b39 the grader correctly rejected them twice and the agent abstained.
  Cause: the rewrite vocabulary map had no entry for "total costs", and the grader did not check that the figure was
  the one asked for.
- **b42** ("which fund's KID is addressed to investors in Switzerland"): retrieval found the right chunk (the Xtrackers
  title "Key Information Document for Investors in Switzerland"), but the grader rejected it because the other funds
  were missing. Cause: the "every fund must appear" rule is right for rankings of figures and wrong for a property
  lookup. The question is also somewhat ambiguous (every KID mentions a Swiss representative); the title is the only
  literal match.

### Fix and how it is checked
Prompt-only changes in `agent/graph.py`: a rewrite mapping for total costs and entry/exit fees plus a rule to keep
distinguishing words; two grader rules (the passages must contain the same kind of figure as asked; property lookups
need only the fund that has the property). Because the fix was designed after seeing these failures, re-running the 52
questions measures regression, not generalisation. Generalisation is checked on a separate 14-question holdout
(`data/golden/holdout_set.jsonl`: 12 cross-document questions of the same two kinds plus 2 traps), written before the
fixed agent was run on it. Results will be added here with the model stated.
