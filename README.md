# Semantic Memory Steward (SMS)

Semantic Memory Steward is a data-governance agent that observes, understands, scores, and relates files to recommend safe actions or require human approval.

## Current MVP Status

This is the Day 1 MVP, featuring:
1. **Semantic Analysis:** A Strands Agent for understanding synthetic files. *(Note: Bedrock integration is currently blocked by an AWS account verification hold)*
2. **Policy Engine:** A deterministic offline rules engine to evaluate semantic results and recommend safe actions (KEEP, ARCHIVE, REVIEW, TRASH).
3. **Observe Layer (S3 Inventory):** A live AWS S3 inventory scanner that maps real S3 objects to deterministic FileMetadata compatible with the Policy Engine.
4. **Observe Layer (S3 Content):** A live AWS S3 content reader that securely fetches and decodes text objects (max 100KB) to prepare for semantic analysis.

**Security and Cost Guardrails:**
- **Bedrock Integration:** The SMS `agent.py` uses the Strands SDK to connect directly to Amazon Bedrock (default: `amazon.nova-micro-v1:0` in `us-east-1`). *Current Status: Bedrock inference is failing with an account-level `Operation not allowed` restriction (AWS Support Case #178868592800270). The offline domain architecture remains fully testable without it.*
- **S3 Usage:** Connects to a dedicated development bucket containing ONLY synthetic fictional files.
- **No Heavy AWS Resources:** Running this codebase does NOT create heavy AWS infrastructure (DynamoDB, Lambda, etc.).
- **Credentials:** AWS Credentials must ALWAYS come from the configured AWS CLI/profile or environment variables. **Never hardcode credentials in source code.**
- **Synthetic Data Only:** The MVP is designed to test with synthetic, non-sensitive data only.
- **Safe Actions:** Risky or destructive actions must eventually require human approval.

## Running Locally

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .[dev]
   ```
2. Run the offline Day 1 demo (runs entirely locally, NO Bedrock calls are made):
   ```bash
   python -m src.sms_agent
   ```
3. Run the S3 Observation layer against your configured bucket:
   ```bash
   # Set SMS_S3_BUCKET environment variable to your development bucket first
   python -m src.sms_agent s3-inventory
   ```
4. Read an object's content safely from S3:
   ```bash
   python -m src.sms_agent s3-read demo/project-plan.txt
   ```
5. Test the End-to-End Orchestration (Stops before Bedrock inference):
   ```bash
   python -m src.sms_agent pipeline demo/project-plan.txt
   ```

## Running Tests

```bash
pytest
```
