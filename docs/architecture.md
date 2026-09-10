# SMS Architecture

## System Overview

Semantic Memory Steward (SMS) is a data-governance agent that autonomously scans an S3 bucket, semantically classifies each document, applies a deterministic safety policy, and either acts autonomously or pauses for human approval — depending on the risk level.

The architecture deliberately separates:

- **AWS-native deterministic layers** (S3, Comprehend, DynamoDB, S3 Vectors, ActionEngine)
- **Semantic analysis layer** (Strands Agent wrapping an LLM provider)
- **Human approval boundary** (enforced before any irreversible action)

---

## Current Architecture (Mermaid)

```mermaid
flowchart TD
    S3_BUCKET["Amazon S3\n(Application Bucket)"]

    subgraph OBSERVE["Observe Layer (AWS-native)"]
        INVENTORY["S3 Inventory\ns3_inventory.py"]
        READER["S3 Content Reader\ns3_content.py"]
    end

    subgraph AGENT["Strands Agent Layer"]
        STRANDS["SMSAgent\nagent.py\n(Strands SDK)"]
        LLM["LLM Provider\n⚠ External fallback active\n(AWS Bedrock / SageMaker\nrestricted — see disclosure)"]
        COMPREHEND["Amazon Comprehend\ncomprehend.py\nEntity + PII detection"]
    end

    subgraph REASON["Reasoning Layer (deterministic)"]
        IMPORTANCE["Importance Scorer\nimportance.py"]
        RELATIONSHIPS["Relationship Analyzer\nrelationships.py"]
    end

    subgraph GOVERNANCE["Governance Layer (deterministic)"]
        POLICY["Policy Engine\npolicy.py\n(KEEP / ARCHIVE / REVIEW / TRASH)"]
        APPROVAL{"Human Approval\nRequired?\n(REVIEW path)"}
    end

    subgraph MEMORY["Semantic Memory (AWS-native)"]
        DYNAMO["Amazon DynamoDB\nSemantic metadata\nper-document"]
        S3VEC["Amazon S3 Vectors\n768-dim cosine index\nSemantic embeddings"]
    end

    subgraph ACTION["Action Layer (AWS-native)"]
        AUTH["ActionAuthorizer\nauthorization.py"]
        ENGINE["ActionEngine\nactions.py"]
        S3_MUTATIONS["Amazon S3\nMutations\n(copy → verify → delete)"]
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
    APPROVAL -->|"Yes: REVIEW → UI"| ENGINE

    ENGINE --> AUTH
    AUTH --> S3_MUTATIONS

    STRANDS -->|"Persist analysis"| DYNAMO
    STRANDS -->|"Persist embedding"| S3VEC

    classDef awsnative fill:#FF9900,color:#000,stroke:#c77b00
    classDef strands fill:#4A90D9,color:#fff,stroke:#2c6da3
    classDef deterministic fill:#2ecc71,color:#000,stroke:#1a8a4a
    classDef restricted fill:#e74c3c,color:#fff,stroke:#c0392b
    classDef human fill:#9b59b6,color:#fff,stroke:#7d3c98

    class S3_BUCKET,INVENTORY,READER,DYNAMO,S3VEC,COMPREHEND,S3_MUTATIONS awsnative
    class STRANDS,LLM strands
    class IMPORTANCE,RELATIONSHIPS,POLICY,AUTH,ENGINE deterministic
    class APPROVAL human
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

## LLM Provider Disclosure

> **Important:** SMS's architecture routes semantic analysis through Strands, which supports AWS Bedrock and SageMaker as native model providers. During this submission, the AWS account has two active restrictions:
>
> - **Amazon Bedrock**: `ValidationException: Operation not allowed` at the account level.
> - **Amazon SageMaker**: Service Quota of **0 instances** for all GPU endpoint families (`ml.g5`, `ml.g4dn`, `ml.g6`, `ml.p3`, `ml.p4`) in `us-east-1`.
>
> As a result, semantic analysis currently uses an **external LLM fallback**, clearly disclosed in both the dashboard System Status panel and this document. The governance, Comprehend enrichment, DynamoDB persistence, S3 Vectors semantic memory, and ActionEngine layers are fully AWS-native and verified live.

---

## Data Flow Summary

1. **Scan**: S3 Inventory lists all objects and their metadata.
2. **Read**: S3 Content Reader fetches text content (max 100 KB, type-safe).
3. **Idempotency gate**: Content hash + ETag checked against DynamoDB; cached analyses are reused.
4. **Semantic analysis**: Strands Agent calls the LLM provider → returns structured `SemanticAnalysisResult` (category, sensitivity, importance_score, reasoning).
5. **Comprehend enrichment**: `detect_entities` and `detect_pii_entities` run in parallel. PERSON entities are annotated separately from PII detection — they do not imply each other.
6. **Relationship analysis**: Exact hash/ETag matching first; then S3 Vectors cosine similarity search for semantic siblings.
7. **Importance scoring**: Deterministic weighted formula (recency, sensitivity, relevance, duplicate penalty).
8. **Policy evaluation**: Deterministic rule matrix → `KEEP`, `ARCHIVE`, `REVIEW`, or `TRASH`.
9. **Human approval boundary**: `REVIEW` decisions halt. A human operator must select an action and explicitly confirm via the dashboard before execution proceeds.
10. **Action execution**: `ActionEngine` validates authorization, then executes (copy-then-verify-then-delete for QUARANTINE; no-op for KEEP).
11. **Persistence**: `SemanticMemoryRecord` written to DynamoDB; embedding written to S3 Vectors.
