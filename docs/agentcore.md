# SMS → Strands → Amazon Bedrock AgentCore

## Status

**Live.** An AgentCore Harness (`sms_semantic_analysis`) is provisioned in
`us-east-1`, `amazon.nova-micro-v1:0` runs the Strands classification step
inside it, and SMS invokes it through the `bedrock-agentcore` data plane.
Deployment was performed under explicit human approval with least-privilege
IAM and a guarded, non-mutating live smoke test.

## Decision

Run the SMS classification step inside an **AgentCore Harness** and invoke it
from SMS via the `bedrock-agentcore` data plane. AgentCore Harness is itself a
**Strands-powered managed agent loop**, which matches SMS's existing `strands.Agent`
usage (`src/sms_agent/agent.py`) with zero framework change.

| Option | What it costs to adopt | Fit for SMS |
|--------|------------------------|-------------|
| **Harness** (chosen) | Config only: execution role + `create-harness` + `get-harness` READY. SMS calls `invoke_harness` per file. | SMS stays the orchestrator (inventory, policy, action engine, human review, relationships, embeddings all unchanged). The LLM prose step is delegated to the managed Strands loop. |
| **Runtime** (container) | ARM64 image → ECR, `create-agent-runtime` + endpoint, custom IAM role, network + authorizer config. | Overkill: SMS's loop is already Strands; Runtime is for writing your own loop/graph/protocol. |
| **No AgentCore** | — | Valid fallback; Bedrock + Strands `structured_output` keeps working. |

## What the adapter does (merged on the AgentCore integration branch)

- `src/sms_agent/agentcore.py` — thin module:
  - `build_messages()` → `InvokeHarness` messages (file key, metadata, content, JSON contract).
  - `_stream_to_text()` → unwraps `contentBlockDelta.delta.text`; surfaces streamed
    `validationException` / `internalServerException` / `runtimeClientError` as `AgentCoreError`.
  - `_parse_result_text()` → JSON → `SemanticAnalysisResult` (shared contract with the
    Gemini fallback). Malformed JSON, empty output, and pydantic `ValidationError` are
    wrapped as `AgentCoreError` (honest failure, never a fake result).
  - `analyze_file_with_harness()` → calls `invoke_harness` with a fresh
    `runtimeSessionId` (stateless, matches SMS's per-file call pattern). Botocore
    `ClientError`/network failures are wrapped as `AgentCoreError`.
- `SMSAgent.analyze_file` — `provider == "agentcore"` branch
  (`SMS_LLM_PROVIDER=agentcore`), same seam as the gemini/groq fallbacks.
  Records a truthful `AI_ANALYSIS` trace event naming the harness and model before
  invoking; raises `ValueError` (no silent fallback) on `AgentCoreError`.
  Default provider is unchanged (`bedrock`).
- `tests/test_agentcore_integration.py` — hermetic tests (mocked client,
  no network, no credentials): stream unwrapping, error surfacing, malformed/empty
  JSON, schema mismatch, authz failure, network failure, no-Gemini-fallback,
  fresh session ids, and trace attribution.
- `scripts/smoke_agentcore.py` — manual, opt-in, non-mutating live smoke test
  (never run by CI; requires `--yes`, real credentials, and the harness ARN).

The harness is **never created by SMS**. SMS only calls an existing harness.

## Live resources (provisioned 2026-09-13, account 527557823928, us-east-1)

| Resource | Value |
|----------|-------|
| Execution role | `arn:aws:iam::527557823928:role/sms-agentcore-harness-role` |
| Role trust policy | Principal `bedrock-agentcore.amazonaws.com` + `aws:SourceAccount=527557823928` (confused-deputy) |
| Role inline policy | `sms-agentcore-model-invoke`: `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` on `arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0` **only** |
| Harness | `arn:aws:bedrock-agentcore:us-east-1:527557823928:harness/sms_semantic_analysis-qRj6vqUQo8` |
| Harness model | `bedrockModelConfig`: `amazon.nova-micro-v1:0`, `maxTokens=2048`, `temperature=0.1` |
| Harness tooling | `tools=[]`, `allowedTools=[]`, `skills=[]` (no tools available to the model) |
| Harness memory | `memory={disabled:{}}` (stateless per call) |
| Guardrails | `maxIterations=15`, `timeoutSeconds=120`, truncation sliding window (150 msgs) |
| System prompt | SMS `SYSTEM_PROMPT` + `JSON_CONTRACT` (harness default; SMS overrides per call) |

### Provisioning notes (lessons captured)

1. `create-harness` **does not** accept hyphens in `harnessName`
   (pattern `[a-zA-Z][a-zA-Z0-9_]{0,39}`) → `sms_semantic_analysis`.
2. `model` is a tagged union: must be `{"bedrockModelConfig": {...}}`, not a flat map.
3. `tags` must be a dict (`{"k":"v"}`), not a list of `{key,value}`.
4. **Trust policy**: the `aws:SourceArn` (harness/\*) condition caused harness control's
   role validation to fail (`Role validation failed ... trust policy allows assumption
   by this service`). Resolution: keep `aws:SourceAccount` (the cross-account confused-deputy
   guard) and drop `aws:SourceArn`. The role-scoped `bedrock:*` ARN + `SourceAccount` still
   bound to this account and this role.
5. **Bedrock model ARN has an empty account segment**: `arn:aws:bedrock:us-east-1::foundation-model/...`
   (not `:us-east-1:527557823928:foundation-model/...`). With the correct ARN the harness's
   assumed role successfully called `ConverseStream`.

## Smoke test result (live, 2026-09-13)

`scripts/smoke_agentcore.py --yes` with `SMS_LLM_PROVIDER=agentcore` analyzed
EXACTLY ONE document (`demo/service-config.json`) through `SMSAgent.analyze_file`
(direct call — bypasses the idempotency cache so the harness is genuinely invoked):

- Before: S3 ETag `a4a9d915…`, 28 objects — After: identical (no mutation performed).
- Result: `category=configuration, sensitivity=public, importance_score=0.7,
  confidence=0.9, recommended_action=retain` (valid `SemanticAnalysisResult`).
- Semantic memory (DynamoDB `sms-semantic-memory`, 28 rows) untouched — analysis
  timestamp for the key unchanged; no row count change; no vector write.

## Cost verification

| Item | Observation |
|------|-------------|
| Control-plane harness | Serverless; idle cost ≈ $0 (no runtime between invocations). |
| Per invocation (verified live) | One Nova Micro round-trip via `ConverseStream` + agent-loop/microVM overhead + CloudWatch logs ≈ negligible single-digit cent range; SMS scans ≤ dozens of files. |
| Idle resources | Only the harness definition + role exist; runtime session ends after the invocation (no always-on compute). |
| Teardown | `aws bedrock-agentcore-control delete-harness --harness-id <id>`; `aws iam delete-role-policy` + `aws iam delete-role`. |

## Environment verification (read-only; S3-Vectors service namespace)

The vector-memory S3 bucket `sms-semantic-vectors-527557823928` (referenced by the
app's default `SMS_VECTOR_BUCKET`) **exists and is live** — earlier "absent" claims
for it in this file were **false**. The bucket lives in the **S3-Vectors platform
namespace** (service `s3vectors`), not in the plain-S3 namespace, so ordinary
`aws s3` / `get-bucket-location` lookups cannot see it. Re-verified live today with
read-only, non-mutating calls (nothing created, deleted, or written):

- `aws s3vectors list-vector-buckets --account-id 527557823928 --region us-east-1`
  → bucket present, ARN
  `arn:aws:s3vectors:us-east-1:527557823928:bucket/sms-semantic-vectors-527557823928`.
- Vector index + vectors confirmed present in the S3-Vectors service namespace.
- DynamoDB table `sms-semantic-memory` present (28 rows, vector write path intact).

This finding was added accidentally in the original branch; it is not real drift and
nothing here caused it. No bucket was created or deleted as part of this work.

## Deployment command sequence (reproducible record)

```bash
# 1. Role (trust: bedrock-agentcore.amazonaws.com + aws:SourceAccount only)
aws iam create-role --role-name sms-agentcore-harness-role \
  --assume-role-policy-document file://sms_trust.json
aws iam put-role-policy --role-name sms-agentcore-harness-role \
  --policy-name sms-agentcore-model-invoke \
  --policy-document file://sms_policy.json

# 2. Harness (NAME: no hyphens; model = tagged union; tags = dict)
aws bedrock-agentcore-control create-harness --cli-input-json file://sms_harness.json
aws bedrock-agentcore-control get-harness --harness-id sms_semantic_analysis-XXXX

# 3. Point SMS at the harness (default provider remains bedrock)
#    SMS_LLM_PROVIDER=agentcore  SMS_AGENTCORE_HARNESS_ARN=<arn>
```

## Safety invariants kept intact

- Policy engine, authorization, human approval, ActionEngine verification,
  and idempotency remain in SMS (never delegated to the harness).
- No deletion of S3 buckets/docs, DynamoDB tables/records, indexes, or CF stacks.
- No changes to `D:\SMS\.env`; never touch `D:\atlas\.env`.
- AgentCore never bypasses SMS's `REVIEW`/protected-delete semantics.
- No silent fallback: AgentCore failures surface as honest `ValueError`
  (`AgentCore/Strands inference failed: …`), never as a simulated result.