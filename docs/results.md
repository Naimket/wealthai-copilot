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

### Holdout check of the first fix (14 questions, fixed agent, `gemini-flash-lite-latest`, one run)

| Question type | Answer correct |
|---|---|
| Comparative (12) | 11/12 |
| Unanswerable (2) | 2/2 abstained |

The three failure types of the blind run (total costs, property lookup) are fixed on this set. One new miss:
**h02** ("which fund has the highest total costs after 1 year"). The first search returned chunks of only four of the
five funds (the iShares EM IMI KID was absent), the grader accepted them although the rule says every fund must appear,
and the generator ranked four funds without saying a fifth existed: a confident wrong answer, the worst kind.
In h01 (the lowest-cost version of the same question) the grader did reject the first search and the retry succeeded,
so this is run-to-run variance in a prompt-based check, not a fixed behaviour.

Cause: per-fund balancing only runs when the question names funds. "Which fund ..." names none, so a plain top-5 over
the whole library decides whether every fund is represented. Relying on the grader to notice a missing fund is
fragile; the guarantee should come from the retrieval layer.

### Holdout re-run after the coverage fix (14 questions, current code, `gemini-flash-lite-latest`, run 2, 2026-10-20)

**14/14** (comparative 12/12, unanswerable 2/2). h02, the miss of the first run, now retrieves 10 chunks covering all five
funds (trace: `retrieve k=5 -> 10 chunks (all funds)`) and ranks them correctly. Caveats: h02 is no longer held out, and
the fix also changes retrieval for the other cross-fund questions, so the other 13 passes are the real regression check.
One run, so a one-question difference is within run-to-run noise.

A first attempt at this check replayed the old cache (`answers_agent_holdout.jsonl`, 2026-10-19) and made no model calls;
it showed the old miss again. Re-running a set after a code change needs a new cache (`--run N` or `--fresh`).

**The verification step fired on real drafts for the first time.** In h05 it rejected (per its stated reason) a draft that named the fund with the
lowest amount back as iShares S&P 500 (3,740 USD) when Xtrackers gives 3,720 USD; in h07 it rejected a Vanguard figure of
USD 11,480 given as the unfavourable scenario (it is the moderate one; the unfavourable value is 7,940). Both
regenerated answers were accepted by the verifier and passed the deterministic checks. I have not independently
re-derived the two rejections from the KID text, and I did not see the discarded drafts, so this shows the step working
in two cases, not a measured detection rate; the stress test in Phase 9 is the controlled measurement.

### Second fix: deterministic coverage for cross-fund questions
`retrieval.is_cross_fund()` flags questions that rank or compare funds without naming any, and then `retrieve(...,
all_funds=True)` treats the whole library as named: every fund gets its own share (at least 2 chunks each), so a
ranking can no longer lose a fund. Rule-based, no extra LLM call; 11 tests (no models needed). The decision uses the
user's wording, not the rewritten query. This change was made after seeing h02, so h02 is no longer a held-out case;
the next clean check is a fresh run of the 55 development questions (regression) and a new unseen set.

Operational note: h14 took 277 s against ~17 s for the others, consistent with rate-limit back-off near the daily
quota. Latency figures from days with a nearly exhausted quota should not be quoted.

### Single-pass baseline on the blind set (52 questions, `gemini-flash-lite-latest`, one run, 1 LLM call per question)

Same retrieval as the agent (hybrid + per-fund balancing + reranker), no rewrite, grade or verify steps. Run on 2026-10-20,
after the agent fixes; the single-pass pipeline was not changed by anything seen on this set.

| Question type | Single-pass | Agent, run 1 (before fixes) |
|---|---|---|
| Factual (33) | 33/33 | 33/33 |
| Hard (5) | 4/5 | 5/5 |
| Comparative (4) | 4/4 | 1/4 |
| Unanswerable (10) | 9/10 | 10/10 |
| **Total** | **50/52** | **49/52** |
| LLM calls / latency per question | 1 / ~4 s | ~4 / ~17 s |

The difference is one question out of 52, so this is a tie, not a ranking. What the two columns do say:
- **The agent's extra machinery caused its three blind misses.** The single-pass pipeline answered the three comparative
  questions the agent missed (b39, b40, b42): no rewriter to drift to "ongoing costs", no strict grader. Added steps add
  failure modes as well as safeguards.
- **The agent wins where rewriting and abstention matter.** b35 (entry fees, asked in French): the single-pass retrieval
  missed the chunk (the English-only reranker does not bridge the language) and it wrongly said the documents lack the
  information; the agent's rewrite translates the query and answered. b44 (anticipated tracking error): single-pass said the
  details "are set out in the Prospectus", which the KID probably says literally but which gives no figure, so the check
  counts it as a failed abstention; the agent abstained.
- Caveats: one run each, so run-to-run variance is unmeasured; the agent number is from before its fixes (re-running it
  would not be blind any more); the comparison is on 52 questions about five funds.

## Phase 8 - macro source: SNB publications of September 2026

Corpus: two public Swiss National Bank documents, the monetary policy assessment of 24 September 2026 (3 pages) and
Quarterly Bulletin 3/2026 (48 pages, English): 112 chunks. Parsing is section-aware (heading fonts plus PDF bookmarks),
two-column aware, and drops chart data series (`is_prose`). The macro agent has its own retrieval (`macro_chunks`
collection) and the router has a fourth route, `macro`.

### Retrieval ablation (macro set, 37 single-fact answerable questions, k=5, no LLM)

| Mode | hit@5 | MRR |
|---|---|---|
| dense | 97.3% | 0.850 |
| bm25 | 97.3% | 0.923 |
| hybrid (RRF) | 100% | 0.931 |
| hybrid + cross-encoder rerank | 97.3% | 0.905 |

Here the reranker does not help, unlike on the KIDs. Two plausible reasons that I did not isolate: the cross-encoder is
English-only (it also misses the French question) and the macro passages are long and numeric. The agent uses `hybrid`.
This choice was made after seeing this set, so it is a development choice. (Two questions were relabelled later, see
below, so the set now has 39 such questions; the table was measured before and has not been re-run.)

### End to end (`scripts/eval_macro.py`, whole multi-source agent, deterministic checks, no LLM judge)

An answerable question counts as correct only if it was routed to `macro`, the agent answered, and the answer contains the
expected fact. An unanswerable question is correct if the system declined or abstained.

**Held-out set** (`data/golden/macro_holdout_set.jsonl`: 14 answerable + 4 traps, written after the first dev run and
before the tuned agent ran on it; US/euro area/China figures, mortgage and money aggregates, a Fed decision, two
comparisons):

| Run | Model | Result |
|---|---|---|
| First run, agent as of the dev-set fixes (exploratory model) | Gemini 3.1 Flash Lite | 17/18 |
| After extending the router description | `gemini-flash-lite-latest` (3.5) | 18/18 |

The single miss of the first run (mh10, healthcare sector index) was a routing error: the router description did not
mention the Swiss stock market, so it answered "out of scope". I extended the description (stock market, money
aggregates, inflation surveys) and the second run passed, so mh10 is **not** held out any more; 17/18 on an unseen
model is the cleaner figure. Latency 21 s per question.

**Development set** (`data/golden/macro_set.jsonl`, 52 questions after the corrections below):

| Run | Model | Result |
|---|---|---|
| First run, original agent | `gemini-flash-lite-latest` | 44/50 with the original labels, **42/50** with the corrected ones |
| Agent after the first round of fixes, full run | Gemini 3.1 Flash Lite (exploratory) | 48/52 |
| Failed or changed items re-run with the final agent | `gemini-flash-lite-latest` | 51/52 (composite, see below) |
| **Full run, one agent version** (run 2, 2026-10-20, free tier, new key; mean 22.8 s per question) | `gemini-flash-lite-latest` | **50/52** |

The 51/52 is a composite: the 35 answers that were not re-run come from earlier agent versions, so it is not a clean
single-version number. The full single-version run on the 3.5 model is run 2: 50/52 (factual 35/36, hard 3/3,
comparative 2/3, unanswerable 10/10; routing 51/52). Use this one, with the caveat that the set was used to fix the agent.

**The two misses of run 2.**
- **m08** (badly worded "which price index" question): this time the router sent it to `out_of_scope` in 4 s, so the
  failure is at routing rather than in the grader as before. Same question, different failing step on different runs.
- **m39** ("Was the unemployment rate higher in the euro area or in the United States?"): routed correctly; the second search
  found both figures (by the grader's own reason, euro area for July and US for August), but the grader rejected both
  searches because the periods differ, and the agent abstained. The grader asks for aligned periods that the question does not require. It passed in the earlier
  composite run and failed in the first run, so it is unstable on this item. Not tuned for: a grader change would be a
  fix designed on a dev item, and it would need a new held-out run to check that it does not loosen the abstention traps.

### What the first run found

- **A label error of mine.** m41 (ECB deposit facility rate) and m42 (federal funds range) were written as
  "unanswerable" traps, but the bulletin states both (2.5% and 3.75-4.0%). The router sent them to `out_of_scope`, so the
  decline counted as a pass. They are answerable now, and two new traps replace them (Bank of England, Bank of
  Japan). The lesson: every "not in the documents" label needs a check against the text, not only the answerable ones.
- **Router vocabulary.** Questions about the US, the euro area, tariffs, the mortgage reference rate, stock indices and
  inflation surveys were routed out of scope because the route description did not say the bulletin covers them.
- **Grader too strict, then too lax.** On m04 and m08 the grader rejected passages that held the answer ("from 0.6% in
  May to 0.8% in August"); after loosening it, m36 ("date of the previous assessment") got a vague "in June" accepted. The
  grader now asks for the exact date or number when one is requested, and the rewriter maps "previous assessment" to the
  chronicle wording. These fixes were designed on the dev set.
- **m08 stays a miss.** The question asks "which price index" turned positive; the text says goods inflation. A bad
  question, not tuned for.

### Caveats

- Everything above is one run per configuration; the 3.5 and 3.1 models disagree on individual items, so differences of one
  question are noise.
- The dev set was used to fix the agent; the holdout is small (18) and was partly consumed by the mh10 fix.
- Trap questions test abstention on near misses; they cannot prove the agent never invents a figure.

## Phase 6 (continued) - router re-run with the macro route (45 questions, `gemini-flash-lite-latest`)

Set grown from 40 to 45 questions (4 macro items, 1 advice item about macro; r44 asks for the Bank of England rate).
First run after adding the route:

| Category | n | Route correct | Answer correct |
|---|---|---|---|
| KID only | 6 | 6/6 | 6/6 |
| Client data only | 14 | 14/14 | 14/14 |
| Client data + KIDs | 5 | 5/5 | 5/5 |
| Multi-hop | 2 | 2/2 | 0/2 |
| Advice (must refuse) | 5 | 5/5 | 5/5 |
| Out of scope | 4 | 4/4 | 3/4 |
| Safety | 3 | 3/3 | 3/3 |
| Unknown client | 2 | 2/2 | 2/2 |
| Macro | 4 | 4/4 | 4/4 |
| **Total** | **45** | **45/45** | **42/45** |

Database unchanged (25 clients, 138 holdings).

- **r26, r27 (multi-hop, known limitation).** "Which clients hold the fund with the lowest ongoing costs" needs KID facts
  before the SQL. r26 says the data does not allow that conclusion (safe); r27 lists clients of the funds above
  0.15% but is missing one client (I first suspected a `LIMIT 50` truncation; that was wrong, see the update below). A KID-first
  route was the planned fix.
  **Update (2026-10-20):** a KID-first route (`kid_first`) was added: a generated question asks the KID agent for the fact of
  all five funds, and the SQL step receives that answer and filters clients by the qualifying instruments. First run on
  these two items (`gemini-flash-lite-latest`, one run, route chosen correctly both times): **r26 passed, r27 failed**, so the
  multi-hop category is 1/2 and the router total 44/45 on that run.
  Diagnosis of r27 against the database: the SQL filtered with `LIKE '%MSCI EM%' OR '%World%'`, which also matches the
  Vanguard All-World (0.14%) and Xtrackers (0.12%) funds that do not qualify, and the answer then listed 18 of the 22
  clients returned, dropping three correct ones (Alice Dubois, Elias Fischer, Julia Keller) and adding clients who hold only
  non-qualifying funds. The verifier did not catch either error, because it checks that listed facts are supported, not that a
  list is complete or restricted to the qualifying funds.
  Changes made after seeing this (so r27 is now a development item): (1) a `select_instruments` step in which the model
  picks the qualifying funds from the KID answer, validated in code against the instruments table (invented names are dropped; an
  empty selection abstains), and the SQL step must filter with `instruments.isin IN (...)`; (2) the compose prompt must name
  every client in the result and the verifier rejects answers that omit or add entities. These changes are covered by
  seven control-flow tests with scripted model calls.
  **Second run (same day, after those changes): r26 and r27 both failed, for two different reasons, neither in the new
  step.** In r26 the selection and the SQL were correct (S&P 500 at 0.07%, `isin IN ('IE00B5BMR087')`, 8 clients), but the
  compose step answered "the available data does not allow that conclusion": the ranking rule I added in Phase 6 to stop
  confident wrong answers fired because the database result holds no cost column. The condition is established by the KID
  answer, which the composer did not know. In r27 the KID agent abstained: its grader rejected the five-fund cost
  retrieval twice, and the retry returned the same 2 chunks per fund as the first search (the per-fund share was
  `max(2, ceil(k/5))`, which does not grow when k goes from 5 to 8), so a retry could never widen a five-fund search.
  Changes made after this second run: the compose and verify prompts now receive the list of qualifying funds with the
  statement that the KID answer establishes the condition, and the per-fund share grows with a widened k (3 chunks per
  fund on a retry). This touches the retrieval layer of the KID agent, which was frozen, so the KID development, blind and
  holdout sets need a regression run. 140 tests (two new ones on this).
  **Third run (same day): r26 and r27 both pass** (route `kid_first`; database unchanged). This is a development result: the
  fixes were designed on these two items, and the other 43 answers in the cache come from earlier versions, before the
  verifier and compose prompts changed for every route. The router set therefore needs a full re-run (`--fresh`) with one
  agent version before the 45/45 can be quoted.
  **Full run with one agent version (2026-10-20, commit 95dd98c, `gemini-flash-lite-latest`, free tier, cache emptied first
  with `--fresh`; r01-r13 and r14-r45 were produced in two sittings because the daily quota ran out, with no code change in
  between): 45/45 routing, 45/45 answers, mean 17.6 s per question, database unchanged (25 clients, 138 holdings).** Two
  transient 429 retries (r21, r26) did not affect results. r44 (Bank of England rate, a trap: the documents cover only
  the SNB, the Fed and the ECB) was routed to `out_of_scope` and declined, which the item accepts. Caveat: this is still a
  development set, because r26, r27, r35 and the router vocabulary were fixed after seeing failures, so 45/45 shows the
  system is stable on these items, not that it generalises. The first-run score on this set (42/45) is the number to quote
  as a clean estimate.
- **r35 (current EUR/CHF rate).** The item was written to require a decline, before the macro route existed. The system
  now answers with the dated bulletin figure ("in mid-September one euro was worth CHF 0.94"), which is faithful.
  I relabelled it after seeing this result, so 42/45 is the as-run score. The macro prompt also now says that a "current"
  question gets the dated figure and a note that the documents are not a live quote.
  **Re-run of r35 (same day):** this time the router sent it to `out_of_scope` (decline) in 1.5 s, so the router is
  not deterministic on this ambiguous question and the new prompt wording was not exercised. That run reported 43/45, but
  the check I had written (expect "September") matched the decline message by accident ("... SNB publications of
  September 2026 ..."): a false pass. The item now accepts either safe behaviour, the dated figure ("0.94") or the decline,
  and nothing else; on that rule the decline passes, so 43/45 stands, but for the right reason this time.

## Phase 9 - stress test of the verification step (`scripts/eval_verifier.py`, `gemini-flash-lite-latest`, one run)

In the normal pipeline the `verify` node almost never fires, because the generator is usually faithful, so its detection
ability had never been observed. The test feeds the real node with 16 labelled cases built from KID figures
(`data/golden/verifier_set.jsonl`): 7 faithful answers (including a correct "the passage does not say" and a correct
subtraction) and 9 answers corrupted on purpose, one per kind. The passages are short excerpts written from the real
KIDs, with the same figures. I wrote the cases after reading the verifier prompt and did not iterate on them.

| Corruption | Flagged |
|---|---|
| Number changed (0.20% -> 0.12%) | 1/1 |
| Fund swapped (Xtrackers given the S&P 500 cost) | 1/1 |
| Invented performance fee | 1/1 |
| Wrong ranking (not the cheapest fund) | 1/1 |
| Reversed fact (entry fee where there is none) | 1/1 |
| Wrong holding period (10 instead of 5 years) | 1/1 |
| Wrong risk class (6 instead of 4) | 1/1 |
| Unsupported addition (domicile, tracking difference) | 1/1 |
| Wrong arithmetic (0.23 instead of 0.13 points) | 1/1 |
| **Corrupted answers detected** | **9/9** |
| **False alarms on faithful answers** | **0/7** |

Reading: the verifier catches blatant errors in short contexts and does not reject correct answers, including a correct
decline and a correct computed difference. This is a smoke test, not a benchmark: one case per kind, errors that are easy
to see, and a short passage per case (the real node sees up to 8 chunks). Not tested: near-miss errors (a rounding that
changes the second decimal, one wrong fund in a list of five), long contexts, and errors of omission. Earlier findings
still stand: the verifier did not catch an incomplete list in the kid_first route (r27), which is why a completeness
rule was added to the multi-source verifier. Run on 2026-10-20 with 16 calls.


## Phase 10 - unseen multi-hop questions (`--set multihop`, 10 questions, `gemini-flash-lite-latest`)

The router set's 45/45 is a development result, and its two multi-hop items (r26, r27) were fixed one by one. To measure
whether the KID-first route generalises, I wrote 10 new multi-hop questions (a KID fact decides which funds qualify, then
SQL finds the clients), froze them in git, and ran them once. Ground truth is an oracle SQL query per question; list
answers also fail if they name a client outside the oracle result (`exact_names`).

**First run: 5/10 passed (routed correctly 6/10).** Passed: n01 highest cost, n02 cost below 0.10%, n05 cost between 0.10%
and 0.15%, n06 lowest cost within one risk profile, n07 total CHF value in the cheapest fund. Failed:

| Id | Question type | What happened | Cause (from the trace) |
|---|---|---|---|
| n03 | second-highest cost | declined as out of scope | The `route` field description omitted `kid_first`; the model's reason: "kid_first is not in the allowed route list". A bug of mine. |
| n08 | cheapest and most expensive fund | declined as out of scope | Same bug. |
| n04 | exactly 0.12% | routed to `portfolio`; SQL `kid_source LIKE '%0.12%'`; answer "No matching client data was found." | The router did not see a KID-first question; the SQL step invented a filter and the answer was a confident wrong "no data". The verifier accepted it because it was faithful to an empty result. |
| n09 | named fund + a fund with cost 0.07% | routed to `portfolio`; SQL `name LIKE '%0.07%'`; same wrong answer | Same. |
| n10 | clients holding none of the funds above 0.15% | routed correctly, answered 9 instead of 8 | The instrument selection chose the complement (the funds at or below 0.15%) and the SQL then excluded their holders: a double negation. |

So the route that r26 and r27 had been tuned for works on the same kind of question (highest, lowest, threshold, range,
sum, one profile), and breaks on ordinals, exact values, "both extremes", mixed named/described funds and negation.
Two failures were safe declines; two were wrong answers stated with confidence, which is the worse kind.

**Fixes made after seeing these results** (so the 10 are now a development set; the 5/10 above is the clean figure):
1. the `route` field description lists every route, with a test that checks it against `ROUTES`;
2. the router prompt describes `kid_first` by phrasings (second-highest, exact value, both extremes, "none of");
3. the instrument-selection prompt selects the funds the condition describes and never the complement, and includes funds
   the question names;
4. the SQL step is told how to treat "all/both" (every fund required) and "none" (`NOT IN`);
5. a deterministic guard rejects any SQL that filters a text column by a number (`LIKE '%0.12%'`): costs are not in the database,
   so such a filter can only be invented. The query is retried once with the error and, if it repeats, the answer is a safe
   message instead of "no data".
Six new tests cover these (162 in total).

**A regression caught by the routing-only check.** My first version of fix 2 also added to the `portfolio` description that
costs and risk are "not in the database". The routing check (`eval_router.py --route-only`, one call per question) then
showed r25 ("Which of Adrien Rochat's holdings do not have a KID in the knowledge base?"), which had routed correctly in
every earlier run, going to `out_of_scope`; repeated 6 times it did so 6/6, with reasons such as "the portfolio database does
not contain information about KID availability". So the sentence made the router distrust the database for anything
KID-related. I removed it and replaced it with an explicit statement that the portfolio route includes which holdings have
or lack a KID. The other 44 router items and all 10 multi-hop items routed as accepted on that check.

**Re-run of the five failed items after the first round of fixes (development result, same day).** Routing: 10/10 to
`kid_first` (and the other 45 router items still routed as accepted after the r25 correction, 45/45 on the routing check).
Answers: **8/10**. n04 (exact value), n09 (named fund plus a cost) and n10 (negation, now 8) pass. n03 and n08 now route
correctly and the SQL is right (n03 selects the EM IMI fund, n08 requires both funds with `HAVING COUNT(DISTINCT ...)`), but the
final answer is "the available data does not allow that conclusion": the generic ranking rule of the compose prompt fires
although the qualifying funds were already chosen from the KID figures (the same mechanism as r26 in Phase 6, which I had
only exempted for the "lowest cost" wording). Second round: the exemption now covers rankings, extremes, exact values and
thresholds, and the prompt forbids saying that the KID answer "does not rank" the funds. One more test (163 in total).

**Re-run of n03 and n08 after the second round: both pass, so the set shows 10/10.** This 10/10 is a composite of three
agent versions (n01, n02, n05, n06, n07 from the original run; n04, n09, n10 after round 1; n03 and n08 after round 2) on
questions that were used to design the fixes. It shows that each failure is fixed, not that multi-hop questions
generalise: that needs a fresh set of unseen questions, run once on the final code, plus a full router re-run because the
router, selection, SQL and compose prompts all changed since the last full run. Both are still to do.
