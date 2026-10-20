# Cloud deployment notes (AWS / GCP)

> **Status: design notes only. Nothing in this document has been deployed or tested in the cloud.**
> The project runs locally (Docker Compose, or `uv run`). This page describes how each local component would map
> to a managed service, what code would have to change, and what a Swiss private-banking context adds.

## 1. What runs today

| Component | Local implementation | State |
|---|---|---|
| API | FastAPI (`src/wealthai/api.py`), Docker image | stateless |
| UI | Streamlit (`app/streamlit_app.py`), second container | stateless |
| Vector index | Qdrant in local mode (files under `data/`) + own BM25 | read-mostly, rebuilt by `ingest*.py` |
| Embeddings / reranker | `fastembed` (bge-small, cross-encoder), CPU, in-process | model files cached on disk |
| Portfolio data | SQLite `data/processed/portfolios.db`, queried read-only by the SQL tool | synthetic, read-only |
| LLM | Gemini API through `wealthai/llm.py` (one provider, free tier) | external |
| CI | GitHub Actions (tests, no LLM calls) | – |

## 2. Target mapping

| Concern | AWS | GCP | Notes |
|---|---|---|---|
| API + UI containers | ECS Fargate (or App Runner) behind an ALB | Cloud Run | Both are stateless; scale to zero is possible, but the embedding/reranker models make cold starts slow, so keep a minimum of one instance. |
| Image registry, CI/CD | ECR; GitHub Actions with OIDC role | Artifact Registry; GitHub Actions with Workload Identity Federation | No long-lived keys in GitHub. |
| Vector store | Qdrant (self-hosted on Fargate with EFS/EBS, or Qdrant Cloud); alternatives: OpenSearch (native hybrid BM25 + vector), pgvector on RDS | Qdrant Cloud or GKE; alternatives: AlloyDB/Cloud SQL pgvector, Vertex AI Vector Search | The retrieval layer talks to a small interface (`retrieval.py`); swapping the store means re-implementing dense search and keeping RRF + rerank unchanged. |
| Portfolio database | RDS/Aurora PostgreSQL | Cloud SQL / AlloyDB PostgreSQL | The SQL tool must connect with a **read-only database role** (see section 3). |
| Document storage | S3 (KIDs, SNB PDFs, processed chunks) | Cloud Storage | Ingestion as a batch job: ECS scheduled task or Lambda / Cloud Run Job. Re-index on new document versions, never in the request path. |
| LLM | Amazon Bedrock (Claude) or direct API | Vertex AI (Gemini, Claude) or direct API | Using the cloud-native endpoint keeps traffic inside the provider's network and uses IAM instead of API keys. |
| Secrets | Secrets Manager | Secret Manager | Replaces the local `.env`. |
| Logs and traces | CloudWatch Logs; the agent's `trace` list is already structured | Cloud Logging | One log record per question: route, retrieval attempts, grader verdicts, verifier verdict, latency. No client names in logs. |
| Scheduled evaluation | EventBridge → ECS task | Cloud Scheduler → Cloud Run Job | Runs the golden sets nightly, stores the metrics, alerts on regression. |

## 3. Code changes this would require

1. **LLM provider abstraction.** `wealthai/llm.py` is Gemini-only. A `generate(prompt, schema)` interface with
   one implementation per provider (Gemini, Bedrock, Anthropic API) is needed. Prompts would need re-evaluation on
   each provider: the numbers in `docs/results.md` are valid for `gemini-flash-lite-latest` only.
2. **Configuration.** The model name, the throttle interval and the API key come from environment variables; the
   Qdrant and database paths are constants in the code. Move everything to one settings object.
3. **Database access.** The SQL tool's safeguards today are SQLite-specific (read-only URI, `query_only`, a SQLite
   authorizer limiting tables and functions). On PostgreSQL they must be rebuilt with database mechanisms: a role
   with `SELECT` only on the three tables (or on views), a statement timeout, a row limit, and the single-statement check.
   They would need their own tests.
4. **Index lifecycle.** Version the index (document set + embedding model + chunking parameters) and store the
   version in every answer's metadata, so an answer can be traced to the exact corpus it used.
5. **Rate limiting and retries.** `MIN_INTERVAL` throttling is a free-tier workaround; in production replace it with
   backoff on 429/503 and a per-user quota at the API layer.

## 4. Swiss private-banking considerations

- **Data residency.** Both providers have a Zurich region (AWS `eu-central-2`, GCP `europe-west6`). Whether a given
  LLM model is served from that region depends on the model and must be checked per provider; it cannot be assumed.
- **Client data.** The demo uses synthetic portfolios. With real client data the LLM would receive data derived from
  client records: pseudonymise identifiers before prompting, send only the columns the question needs, and check the
  provider's retention and training terms. Client confidentiality (banking secrecy) and FADP/GDPR apply.
- **Outsourcing.** Using a cloud provider for client-related processing is, for a bank, an outsourcing question
  (FINMA circular on outsourcing). It is a compliance decision, not a technical one.
- **Advice boundary.** The agent already refuses personalised-advice requests (`advice` route) and is a retrieval
  assistant for the *advisor*, not a client-facing recommender. That boundary should be kept and tested.
- **Auditability.** Keep the question, retrieved passages, answer and verifier verdict for each response, with
  access controls, so an advisor's use of the assistant can be reviewed.

## 5. What should be measured before calling it production-ready

- Latency per route under load. The agent makes several LLM calls per question; locally on the free tier the mean is
  about 16–17 s per question (up to ~40 s when two sources are used), partly because of the request throttle.
- Cost per question on the chosen provider (count calls per route; the local call counter on the roadmap supplies this).
- Regression results on all golden sets after every change of model, prompt or index.
- Behaviour when the LLM provider is unavailable (the current agent raises; it should degrade to a clear message).
