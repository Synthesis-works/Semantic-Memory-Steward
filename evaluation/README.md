# SMS Evaluation Harness

Semantic Memory Steward (SMS) leverages both **deterministic governance** and **LLM-powered semantic analysis**. 

This evaluation harness is designed to verify that the agent's semantic understanding and the strict safety invariants of the deterministic pipeline correctly map to desired governance outcomes.

## Architecture

The evaluation suite operates in two modes:

1. **Offline Mode (`mode="offline"`)**: 
   - Never calls the LLM (bypasses Bedrock/Strands).
   - Verifies the **deterministic governance invariants**: relationships, importance scoring, and the `PolicyEngine`'s human-in-the-loop and destructive-action safeguards.
   - Ideal for continuous integration and deterministic safety regression.

2. **Live Mode (`mode="live"`)**:
   - Calls the real Strands LLM integration via Amazon Bedrock (or alternative providers).
   - Evaluates **semantic classification accuracy** against the ground truth.
   - Strictly fails if the LLM provider is blocked/unavailable. **NEVER** silently falls back to a mocked or faked response.

## Synthetic Cases Evaluated

We evaluate against representative professional documents (found in the `demo/` S3 prefix):

| Document | Semantic Expectation | Governance Invariant |
| --- | --- | --- |
| `project-plan.txt` | project/planning, high importance, low sensitivity | KEEP |
| `old-project-log.txt` | history/stale, low importance | ARCHIVE |
| `employee-contacts.txt` | HR/PII, sensitive | REVIEW (No autonomous destruction) |
| `financial-report.txt` | Finance, sensitive/important | REVIEW (No autonomous destruction) |
| `financial-report.txt` (Duplicate) | Duplicate relationship detected | REVIEW (Duplicate detection does not override safety invariants) |

## Key Invariants

The evaluation harness distinguishes between **semantic correctness** (did the LLM accurately guess the document's category?) and **governance invariants** (did the system prevent autonomous destruction of a sensitive document?).

Governance invariants are treated as hard safety rules and should *never* depend solely on the correctness of the LLM.
