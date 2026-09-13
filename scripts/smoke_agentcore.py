"""Manual, opt-in live smoke test for the AgentCore Harness provider.

Analyzes EXACTLY ONE S3 document through SMSAgent.analyze_file with
SMS_LLM_PROVIDER=agentcore, then proves no mutation occurred. This script
is never run by CI and never executes actions (no ActionEngine/MutationEngine).

Design rules honored here:
- Requires an explicit --yes flag (no accidental live invocations).
- Requires SMS_AGENTCORE_HARNESS_ARN and real AWS credentials.
- Calls SMSAgent.analyze_file directly so the idempotency cache cannot
  short-circuit the analysis: the harness is genuinely invoked.
- Records the S3 object (ETag + object count) before and after and fails
  if anything changed.

Usage:
    set SMS_LLM_PROVIDER=agentcore
    set SMS_AGENTCORE_HARNESS_ARN=arn:aws:bedrock-agentcore:us-east-1:<acct>:harness/<id>
    .venv\\Scripts\\python.exe scripts\\smoke_agentcore.py --yes
"""
import argparse
import os
import sys

import boto3

BUCKET_ENV = "SMS_S3_BUCKET"
HARNESS_ENV = "SMS_AGENTCORE_HARNESS_ARN"
DEFAULT_KEY = "demo/service-config.json"
DEFAULT_BUCKET = "semantic-memory-steward-dev-527557823928"


def _require_creds() -> None:
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        sys.exit(
            "FATAL: no AWS credentials found. Configure a profile/role before running."
        )


def _object_inventory(s3, bucket: str, key: str):
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        etag = head.get("ETag")
        metadata = head.get("Metadata", {})
    except boto3.exceptions.BotoCoreError as exc:
        sys.exit(f"FATAL: could not read {bucket}/{key}: {exc}")
    except Exception as exc:  # botocore ClientError etc.
        sys.exit(f"FATAL: could not read {bucket}/{key}: {exc}")
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return {"etag": etag, "metadata": metadata, "object_count": len(keys)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="acknowledge live invocation")
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"S3 key to analyze (default {DEFAULT_KEY})",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv(BUCKET_ENV, DEFAULT_BUCKET),
        help="S3 bucket (default from SMS_S3_BUCKET or the dev bucket)",
    )
    args = parser.parse_args()

    if not args.yes:
        sys.exit(
            "ERROR: live smoke test requires --yes (it invokes a real AgentCore harness once)."
        )
    if os.getenv("SMS_LLM_PROVIDER", "bedrock").strip().lower() != "agentcore":
        sys.exit("ERROR: set SMS_LLM_PROVIDER=agentcore for this smoke test.")
    harness_arn = os.getenv(HARNESS_ENV)
    if not harness_arn:
        sys.exit(f"ERROR: {HARNESS_ENV} must be set.")
    if not args.key.strip() or args.key.startswith("/") or ".." in args.key:
        sys.exit(f"ERROR: unsafe key {args.key!r} not allowed.")

    _require_creds()

    s3 = boto3.client("s3")
    before = _object_inventory(s3, args.bucket, args.key)
    print(f"[smoke] before: {args.bucket}/{args.key} etag={before['etag']} "
          f"objects={before['object_count']}")

    from sms_agent.agent import SMSAgent

    print(f"[smoke] invoking AgentCore harness: {harness_arn}")
    agent = SMSAgent()
    try:
        content = s3.get_object(Bucket=args.bucket, Key=args.key)["Body"].read().decode(
            "utf-8", errors="replace"
        )
    except Exception as exc:
        sys.exit(f"FATAL: could not fetch {args.bucket}/{args.key}: {exc}")

    try:
        result = agent.analyze_file(args.key, content)
    except Exception as exc:
        print(f"[smoke] FAIL: AgentCore analysis errored honestly ({type(exc).__name__}): {exc}")
        return 1

    after = _object_inventory(s3, args.bucket, args.key)
    print(f"[smoke] after:  {args.bucket}/{args.key} etag={after['etag']} "
          f"objects={after['object_count']}")

    if before != after:
        print("[smoke] FAIL: S3 inventory changed during analysis (mutation detected).")
        return 1

    print("[smoke] S3 inventory unchanged: no mutation performed.")
    print(f"[smoke] provider={type(agent).__name__} model={agent.model_id}")
    print("[smoke] RESULT:")
    for field in ("key", "category", "sensitivity", "importance_score",
                  "confidence", "recommended_action", "reasoning"):
        print(f"  {field}: {getattr(result, field)}")
    print("[smoke] SUCCESS: AgentCore harness returned a valid SemanticAnalysisResult.")
    return 0


if __name__ == "__main__":
    sys.exit(main())