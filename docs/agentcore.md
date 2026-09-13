# SMS → Strands → Amazon Bedrock AgentCore

## Status

**Investigation + adapter code only. No AWS resource has been created.**
Deployment of any AgentCore resource requires explicit human approval.

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

## What the adapter does (merged into `feature/agentic-ux`)

- `src/sms_agent/agentcore.py` — new, thin module:
  - `build_messages()` → `InvokeHarness` messages (file key, metadata, content, JSON contract).
  - `_stream_to_text()` → unwraps `contentBlockDelta.delta.text`; surfaces streamed
    `validationException` / `internalServerException` / `runtimeClientError` as `AgentCoreError`.
  - `_parse_result_text()` → JSON → `SemanticAnalysisResult` (same contract as the
    Gemini fallback, including the shared `recommended_action` case normalizer).
  - `analyze_file_with_harness()` → calls `invoke_harness` with a fresh
    `runtimeSessionId` (stateless, matches SMS's per-file call pattern).
- `SMSAgent.analyze_file` — new `provider == "agentcore"` branch
  (`SMS_LLM_PROVIDER=agentcore`), same seam as the gemini/groq fallbacks.
  Reads `SMS_AGENTCORE_HARNESS_ARN`. No product-behavior change to the
  default `bedrock` path.
- `tests/test_agentcore_integration.py` — 10 hermetic tests (mocked client,
  no network, no credentials).

The harness is **never created by SMS**. SMS only calls an existing harness.

## Cost and risk assessment

| Item | Assessment |
|------|------------|
| Harness control-plane resource | Serverless; no idle compute. Idle cost ≈ $0 (no runtime running between invocations). |
| Per invocation | Model tokens (Nova Micro `amazon.nova-micro-v1:0`, on-demand, effectively $0.00014–0.00021/1K in tokens) + agent loop overhead (harness tokens/microVM-seconds per session) + CloudWatch logs. SMS analyzes typically ≤ dozens of files per scan → negligible; still, set explicit runtime guardrails. |
| Execution role | New IAM role required — **needs approval** (user rule: no IAM changes without approval). Scope least-privilege (Bedrock model ARNs only; `bedrock-agentcore:*` actions only as needed) and add confused-deputy `aws:SourceAccount` / `aws:SourceArn` conditions. |
| Trust boundary | `invoke_harness` input is trusted only because SMS is the sole caller and SMS's policy/action engine already gates destructive actions. Keep authorizer (SigV4 default) and never accept end-user-supplied override fields. |
| Rate / abuse | Add application-layer throttling in front of SMS's harness calls; each invocation spins an isolated microVM. |
| Logging/PII | Harness observability captures tool I/O and payloads. Encrypt the harness CloudWatch log group with a KMS key and set retention; do not enable verbose tracing until reviewed. |
| Model access | `amazon.nova-micro-v1:0` already authorized in `us-east-1` (account `527557823928`). |

## Deployment command sequence (NOT executed — for approval only)

All commands are dry-run/planning and must not be run without approval.

```bash
# 1. Least-privilege execution role with confused-deputy guard
#    (role trust policy):
#    { "Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},
#      "Action":"sts:AssumeRole",
#      "Condition":{"StringEquals":{"aws:SourceAccount":"527557823928"},
#                   "ArnLike":{"aws:SourceArn":"arn:aws:bedrock-agentcore:us-east-1:527557823928:harness/*"}}}

# 2. Create the harness (control plane). Explicit guardrails, SigV4 default.
aws bedrock-agentcore-control create-harness \
  --harness-name SMSAnalyzer \
  --execution-role-arn ARN_OF_STEP_1_ROLE \
  --model '{"bedrockModelConfig":{"modelId":"amazon.nova-micro-v1:0","maxTokens":1024,"temperature":0.1}}' \
  --max-iterations 15 --max-tokens 2048 --timeout-seconds 120

# 3. Poll until READY (a harness is not invokable before READY).
aws bedrock-agentcore-control get-harness --harness-id <harnessId>

# 4. Point SMS at the harness.
#    export SMS_LLM_PROVIDER=agentcore
#    export SMS_AGENTCORE_HARNESS_ARN=<arn>

# 5. Validate with the existing hermetic suite, then a guarded live test:
#    AWS_PROFILE=opencode AWSCLI-region us-east-1 smoke test on a demo file
#    with execute_action=False (same policy engine, human-approval, verification
#    chain; agent output still flows through SMS ActionEngine verification).
```

If any of steps 1–3 would create a meaningfully billable resource, or the
execution-role policy would exceed least privilege, stop and report.

## Safety invariants kept intact

- Policy engine, authorization, human approval, ActionEngine verification,
  and idempotency remain in SMS (never delegated to the harness).
- No deletion of S3 buckets/docs, DynamoDB tables/records, indexes, or CF stacks.
- No changes to `D:\SMS\.env`; never touch `D:\atlas\.env`.
- AgentCore never bypasses SMS's `REVIEW`/protected-delete semantics.