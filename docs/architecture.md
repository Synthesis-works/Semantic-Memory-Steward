# SMS Architecture

## System Overview

Semantic Memory Steward (SMS) is a data-governance agent that autonomously scans an S3 bucket, semantically classifies each document, applies a deterministic safety policy, and either acts autonomously or pauses for human approval — depending on the risk level.

The architecture deliberately separates:

- **AWS-native deterministic layers** (S3, Comprehend, DynamoDB, S3 Vectors, ActionEngine)
- **Semantic analysis layer** (Strands Agent wrapping an LLM provider, with an AgentCore Harness available)
- **Human approval boundary** (enforced before any irreversible action)

---

## Architecture Diagram

![Semantic Memory Steward architecture](assets/architecture.png)

---

## Current Architecture (Mermaid)

The Mermaid source below is the editable, source-of-truth representation of the architecture diagram.

```mermaid
flowchart TD

    S3_BUCKET["Amazon S3<br/>(Application Bucket)"]

    subgraph OBSERVE["Observe Layer (AWS-native)"]
        INVENTORY["S3 Inventory<br/>s3_inventory.py"]
        READER["S3 Content Reader<br/>s3_content.py"]
    end

    subgraph AGENT["Strands Agent Layer"]
        STRANDS["SMSAgent<br/>agent.py<br/>(Strands SDK)"]
        LLM["LLM Provider<br/>Amazon Bedrock (Nova Micro)<br/>via Strands — AgentCore Harness available"]
        COMPREHEND["Amazon Comprehend<br/>comprehend.py<br/>Entity + PII detection"]
    end

    subgraph REASON["Reasoning Layer (deterministic)"]
        IMPORTANCE["Importance Scorer<br/>importance.py"]
        RELATIONSHIPS["Relationship Analyzer<br/>relationships.py"]
    end

    subgraph GOVERNANCE["Governance Layer (deterministic)"]
        POLICY["Policy Engine<br/>policy.py<br/>(KEEP / ARCHIVE / REVIEW / QUARANTINE)"]
        APPROVAL{"Human Approval<br/>Required?<br/>(REVIEW path)"}
        REVIEW_UI["Human Approval UI<br/>Explicit approval required"]
    end

    subgraph MEMORY["Semantic Memory (AWS-native)"]
        DYNAMO["Amazon DynamoDB<br/>Semantic metadata<br/>per-document"]
        S3VEC["Amazon S3 Vectors<br/>1024-dim cosine index<br/>Semantic embeddings (Titan V2)"]
    end

    subgraph ACTION["Action Layer (AWS-native)"]
        AUTH["ActionAuthorizer<br/>authorization.py"]
        ENGINE["ActionEngine<br/>actions.py"]
        S3_MUTATIONS["Amazon S3<br/>Mutations<br/>(copy → verify → delete)"]
    end

    S3_BUCKET --> INVENTORY
    S3_BUCKET --> READER

    INVENTORY --> STRANDS
    READER --> STRANDS

    STRANDS --> LLM
    STRANDS --> COMPREHEND
    STRANDS --> IMPORTANCE
    STRANDS --> RELATIONSHIPS

    S3VEC -->|"Semantic search"| RELATIONSHIPS
    RELATIONSHIPS --> DYNAMO

    IMPORTANCE --> POLICY
    RELATIONSHIPS --> POLICY
    COMPREHEND --> POLICY

    POLICY --> APPROVAL

    APPROVAL -->|"No: KEEP / ARCHIVE"| ENGINE
    APPROVAL -->|"Yes: REVIEW"| REVIEW_UI
    REVIEW_UI -->|"Approved"| ENGINE

    ENGINE --> AUTH
    AUTH --> S3_MUTATIONS

    STRANDS -->|"Persist analysis"| DYNAMO
    STRANDS -->|"Persist embedding"| S3VEC

    classDef awsnative fill:#FF9900,color:#000,stroke:#c77b00
    classDef strands fill:#4A90D9,color:#fff,stroke:#2c6da3
    classDef deterministic fill:#2ecc71,color:#000,stroke:#1a8a4a
    classDef human fill:#9b59b6,color:#fff,stroke:#7d3c98

    class S3_BUCKET,INVENTORY,READER,DYNAMO,S3VEC,COMPREHEND,S3_MUTATIONS awsnative
    class STRANDS,LLM strands
    class IMPORTANCE,RELATIONSHIPS,POLICY,AUTH,ENGINE deterministic
    class APPROVAL,REVIEW_UI human
```

---

## Legend

| Colour | Meaning |
|--------|---------|
| 🟠 Orange | **AWS-native** — verified live against the AWS account |
| 🔵 Blue | **Strands Agent** — model-provider abstraction layer |
| 🟢 Green | **Deterministic** — no LLM involved; logic is fully auditable |
| 🟣 Purple | **Human approval boundary** — autonomy halts here for REVIEW decisions |

---

## Data Flow Summary

1. **Scan**: S3 Inventory lists all objects and their metadata.
2. **Read**: S3 Content Reader fetches text content (max 100 KB, type-safe).
3. **Idempotency gate**: Content hash + ETag checked against DynamoDB; cached analyses are reused.
4. **Semantic analysis**: Strands Agent calls the LLM provider (Amazon Bedrock — Nova Micro) → returns structured `SemanticAnalysisResult` (category, sensitivity, importance_score, reasoning). An AgentCore Harness is available as an integration for running the agent loop.
5. **Comprehend enrichment**: `detect_entities` and `detect_pii_entities` run in parallel. PERSON entities are annotated separately from PII detection — they do not imply each other.
6. **Relationship analysis**: Exact hash/ETag matching first; then S3 Vectors (1024-dim Titan V2 embeddings) cosine similarity search for semantic siblings.
7. **Importance scoring**: Deterministic weighted formula (recency, sensitivity, relevance, duplicate penalty).
8. **Policy evaluation**: Deterministic rule matrix → `KEEP`, `ARCHIVE`, `REVIEW`, or `QUARANTINE`.
9. **Human approval boundary**: `REVIEW` decisions halt. A human operator must explicitly approve via the Human Approval UI before execution proceeds.
10. **Action execution**: `ActionEngine` validates authorization via the `ActionAuthorizer`, then executes (copy-then-verify-then-delete for QUARANTINE; no-op for KEEP).
11. **Persistence**: `SemanticMemoryRecord` written to DynamoDB; embedding written to S3 Vectors.