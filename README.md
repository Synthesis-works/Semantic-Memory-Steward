# Semantic Memory Steward (SMS)

> **AWS Hackathon Submission** — Built with [Strands Agents SDK](https://strandsagents.com/)

Semantic Memory Steward is a data-governance agent that transforms passive S3 storage into an **active, semantic memory system**. It scans documents, understands their meaning using a Strands-orchestrated LLM, enriches classifications with real-time AWS Comprehend signals, scores their importance, detects relationships and duplicates, and applies a deterministic safety policy — halting autonomy and requiring human approval whenever risk is high.

---

## What SMS Does

Traditional storage systems know a file's **name, size, and date**. SMS knows what a file **means**.

```
S3 documents
     ↓
SMS scans & reads content (AWS-native)
     ↓
Strands Agent → semantic classification
     ↓
Amazon Comprehend → entity & PII enrichment
     ↓
Importance scoring + relationship detection
     ↓
Policy Engine (deterministic safety)
     ↓
┌──────────┬────────────┬──────────┐
│   KEEP   │   ARCHIVE  │  REVIEW  │
└──────────┴─────┬──────┴──────────┘
                 ↓
         Human Approval UI
                 ↓
          Action Engine (AWS-native)
```

---

## Why Strands?

Strands Agents provides a **clean model-provider boundary** — the same SMS pipeline can route to AWS Bedrock, Amazon SageMaker, or an external provider, simply by changing configuration. This means:

- The governance, enrichment, and persistence layers are **independent of the model**.
- Structured output (`SemanticAnalysisResult`) is enforced at the Strands boundary, preventing hallucinated decisions from reaching the action layer.
- The policy engine is fully **deterministic** — the LLM supplies semantic understanding, but it cannot override safety rules.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full Mermaid diagram and data flow.

### AWS-Native Components (Verified Live)

| Component | Service | Purpose |
|-----------|---------|---------|
| Inventory | Amazon S3 | List objects and metadata |
| Content Reader | Amazon S3 | Fetch and decode text content |
| Entity/PII Enrichment | Amazon Comprehend | `detect_entities` + `detect_pii_entities` |
| Semantic Memory | Amazon DynamoDB | Persist per-document classification metadata |
| Vector Memory | Amazon S3 Vectors | 1024-dim cosine index for semantic similarity search |
| Action Engine | Amazon S3 | Copy-then-verify-then-delete for quarantine actions |
| Infrastructure | AWS SAM / CloudFormation | DynamoDB table + S3 Vectors provisioning |

### LLM Provider Status

> **Bedrock is live and verified.** `amazon.nova-micro-v1:0` is authorized in
> `us-east-1` and runs SMS's semantic classification through Strands
> (`SMS_LLM_PROVIDER=bedrock`, the default). An optional **AgentCore Harness**
> provider (`SMS_LLM_PROVIDER=agentcore` + `SMS_AGENTCORE_HARNESS_ARN`) runs the
> same Strands classification step inside a managed harness (Nova Micro), verified
> by a live non-mutating smoke test — see [`docs/agentcore.md`](docs/agentcore.md).
> SageMaker's Service Quota is still 0 GPU instances for endpoint usage in
> `us-east-1`. External (Gemini/Groq/Mistral/NVIDIA) providers remain available
> by explicit choice. The active provider is disclosed in the dashboard System Panel
> and in the execution trace; failures are reported honestly, never simulated.

---

## Safety & Human Approval Model

SMS is built around the principle that **autonomy should be bounded by risk**:

- **KEEP** — No mutation. File stays as-is.
- **ARCHIVE** — Autonomous. Low-risk transition for stale, low-importance content.
- **REVIEW** — **Autonomy halts.** The Streamlit dashboard surfaces the document, the semantic reasoning, and the Comprehend signals. A human operator must explicitly select `KEEP` or `QUARANTINE` and confirm before the ActionEngine executes anything.
- **TRASH → QUARANTINE** — Copy-then-verify-then-delete to `trash/` prefix. Requires human approval for any high-sensitivity document.

Critically: **the ActionEngine hard-blocks** any unapproved destructive action, regardless of what the LLM recommended.

---

## Project Structure

```
SMS/
├── app.py                    # Streamlit dashboard (run this for the demo)
├── template.yaml             # AWS SAM infrastructure (DynamoDB + S3 Vectors)
├── pyproject.toml            # Project metadata and dependencies
├── .env.example              # Environment variable template
├── src/sms_agent/
│   ├── agent.py              # SMSAgent — Strands wrapper, structured output enforcement
│   ├── pipeline.py           # SMSPipeline — full orchestration with idempotency
│   ├── comprehend.py         # Amazon Comprehend enrichment (PERSON ≠ PII)
│   ├── policy.py             # Deterministic PolicyEngine
│   ├── actions.py            # ActionEngine with copy-verify-delete safety
│   ├── authorization.py      # ActionAuthorizer — approval boundary
│   ├── importance.py         # Deterministic importance scoring
│   ├── relationships.py      # Hash/ETag + semantic vector relationship detection
│   ├── memory.py             # DynamoDB + S3 Vectors persistence
│   ├── embeddings.py         # Embedding providers (Gemini, Bedrock)
│   ├── s3_inventory.py       # S3 object inventory
│   ├── s3_content.py         # S3 content reader
│   └── models.py             # Pydantic domain models
├── tests/                    # 158 tests, 100% passing
├── evaluation/               # Offline governance evaluation harness
│   ├── runner.py             # Evaluator (offline + live modes)
│   └── fixtures.py           # 5 test cases covering all policy branches
├── docs/
│   ├── architecture.md       # Architecture diagram (Mermaid)
│   └── demo-scenario.md      # Demo narrative
└── scripts/experiments/      # Historical AWS investigation scripts (not production)
```

---

## Setup

### Prerequisites

- Python 3.9+
- AWS CLI configured with a profile that has access to S3, DynamoDB, Comprehend, and S3 Vectors
- An S3 bucket with demo documents (see below)
- DynamoDB table: `sms-semantic-memory`
- S3 Vectors bucket + index (provisioned via SAM template or manually)

### Installation

```bash
# 1. Clone and set up virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install the package and dev dependencies
pip install -e ".[dev]"

# 3. Install dashboard dependency
pip install streamlit

# 4. Configure environment
copy .env.example .env
# Edit .env with your bucket name, DynamoDB table, S3 Vectors details, and AWS profile
```

### Environment Variables (`.env`)

```bash
SMS_S3_BUCKET=your-bucket-name
SMS_DYNAMO_TABLE=sms-semantic-memory
SMS_VECTOR_BUCKET=your-vectors-bucket
SMS_VECTOR_INDEX=sms-embeddings
AWS_PROFILE=your-profile
AWS_DEFAULT_REGION=us-east-1

# LLM provider: bedrock (default, live) | agentcore | gemini/groq/mistral/nvidia
SMS_LLM_PROVIDER=bedrock
SMS_BEDROCK_MODEL_ID=amazon.nova-micro-v1:0
# Optional AgentCore harness execution (see docs/agentcore.md):
# SMS_LLM_PROVIDER=agentcore
# SMS_AGENTCORE_HARNESS_ARN=arn:aws:bedrock-agentcore:us-east-1:<acct>:harness/<id>
GEMINI_API_KEY=your-key-here
```

---

## Running the Tests

```bash
pytest
```

363 hermetic tests covering: pipeline idempotency, policy invariants, relationship detection, authorization boundaries, Comprehend enrichment logic, DynamoDB/S3 Vectors persistence, action engine safety, human approval boundary, Strands/Bedrock wiring, and the AgentCore harness adapter (no credentials or live AWS calls required).

---

## Running the Streamlit Dashboard

```bash
# Authenticate with AWS first
aws login --profile your-profile

# Launch the dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Demo Walkthrough (4 Documents)

The bucket's `demo/` prefix contains four synthetic documents that exercise every policy branch:

| Document | What SMS Detects | Policy Decision | Why |
|----------|-----------------|----------------|-----|
| `project-plan.txt` | `internal` sensitivity, high importance | **KEEP** | Important internal asset, no risk signals |
| `old-project-log.txt` | `internal` sensitivity, low importance | **ARCHIVE** | Stale, low-value history |
| `employee-contacts.txt` | `confidential`, PERSON entities detected by Comprehend | **REVIEW** | Sensitive HR content, human approval required |
| `financial-report.txt` | `restricted` sensitivity | **REVIEW** | High-risk financial data, human approval required |

### Demo Steps

1. **Open the dashboard** — `streamlit run app.py`
2. **Observe the System Status panel** in the sidebar — AWS component health, honest LLM provider disclosure
3. **Click "▶ Run Full SMS Scan"** — watch each document flow through the pipeline
4. **Click on `employee-contacts.txt`** in the document table — inspect the Document Inspector:
   - AWS Comprehend signals: `Entities: PERSON` and `PII detected: no/yes`
   - Semantic reasoning from the LLM
   - Policy decision: `REVIEW`
5. **Trigger the Approval Queue** — the UI blocks autonomous execution with a red alert
6. **Select `QUARANTINE` → Confirm** — observe the ActionEngine copy-then-delete to `trash/demo/employee-contacts.txt`
7. **Run the scan again** — verify the document no longer appears in the main prefix

---

## Evaluation Harness

SMS includes an offline evaluation suite that tests governance invariants without requiring a live LLM:

```bash
cd evaluation
python runner.py
```

This runs 5 evaluation cases covering all policy branches and verifies:
- Human approval invariant (REVIEW always requires approval)
- Destructive action safety (no autonomous deletion of sensitive content)
- Relationship detection accuracy
- Policy action correctness

---

## Infrastructure

The SAM template (`template.yaml`) provisions:
- **DynamoDB table** (`sms-semantic-memory`) — semantic metadata store
- **S3 Vectors bucket + index** — 768-dimensional cosine similarity index for semantic relationship search

To deploy infrastructure:
```bash
sam build
sam deploy --guided
```
