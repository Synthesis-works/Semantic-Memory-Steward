"""AgentCore Harness adapter for SMS.

SMS owns the product logic (inventory, policy, action engine, human
review, relationships, embeddings). The single prose "classification"
step that today runs over Strands/Bedrock structured_output can instead
run inside an Amazon Bedrock AgentCore Harness — a managed agent loop
that is itself powered by Strands Agents — and be invoked from SMS via
the bedrock-agentcore data plane.

This module is deliberately thin and creates no AWS resources. The
harness must already exist (provisioned only under explicit approval);
SMS just calls it. Provider selection happens in SMSAgent.analyze_file
via SMS_LLM_PROVIDER=agentcore, on the same seam as the gemini/groq
fallbacks, so no product behavior changes.
"""

import json
import os
import uuid
from typing import Dict, List, Optional, Sequence

from .models import SemanticAnalysisResult

HARNESS_ARN_ENV = "SMS_AGENTCORE_HARNESS_ARN"

SYSTEM_PROMPT = (
    "You are Semantic Memory Steward, a professional data-governance agent. "
    "Analyze a file's content and metadata to determine its semantic category, "
    "sensitivity, importance, and a safe recommended action. Be conservative with "
    "sensitive or important information. Never recommend destructive deletion solely "
    "because a file is old or duplicated. Explain the reasoning. "
    "Respond ONLY with a valid JSON object matching the requested schema."
)

JSON_CONTRACT = (
    "Return a single JSON object (no markdown fences, no prose) with exactly these "
    'fields: "category" (string), "sensitivity" (one of public, internal, '
    'confidential, restricted), "importance_score" (number 0-1), "confidence" '
    '(number 0-1), "reasoning" (string), "recommended_action" (one of retain, '
    "archive, review, delete)."
)


class AgentCoreError(RuntimeError):
    """Invocation failed at the AgentCore boundary (streamed error events)."""


def _stream_to_text(stream: Sequence[Dict]) -> str:
    """Unwrap InvokeHarness streamed events into the assistant's text.

    contentBlockDelta.delta.text chunks are concatenated. Streamed error
    events (validationException, internalServerException,
    runtimeClientError) raise AgentCoreError instead of returning partial
    text; everything else (messageStart/messageStop/metadata/toolUse) is
    ignored by design.
    """
    text = ""
    for event in stream or []:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            chunk = delta.get("text") if isinstance(delta, dict) else None
            if isinstance(chunk, str):
                text += chunk
        for error_key in ("validationException", "internalServerException",
                          "runtimeClientError"):
            if error_key not in event:
                continue
            reason = event[error_key]
            message = reason.get("message", "") if isinstance(reason, dict) else ""
            raise AgentCoreError(f"{error_key}: {message or reason}")
    if not text:
        raise AgentCoreError("AgentCore harness stream contained no text content")
    return text


def _parse_result_text(text: str, file_key: str) -> SemanticAnalysisResult:
    """Parse assistant JSON text into SemanticAnalysisResult (gemini-style)."""
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    data = json.loads(clean_text.strip())
    if not isinstance(data, dict):
        raise AgentCoreError("Harness response was not a JSON object")
    data["key"] = file_key
    return SemanticAnalysisResult(**data)


def build_messages(file_key: str, content: str,
                   metadata: Optional[dict] = None) -> List[Dict]:
    """Build the InvokeHarness messages list for one file analysis."""
    metadata_str = json.dumps(metadata) if metadata else "None"
    prompt = (
        f"File Key: {file_key}\n"
        f"Metadata: {metadata_str}\n"
        f"Content:\n{content}\n\n"
        f"Analyze the above file and return the JSON result. {JSON_CONTRACT}"
    )
    return [{"role": "user", "content": [{"text": prompt}]}]


def _default_client(region_name: Optional[str] = None):
    import boto3
    region = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("bedrock-agentcore", region_name=region)


def analyze_file_with_harness(
    file_key: str,
    content: str,
    metadata: Optional[dict] = None,
    *,
    harness_arn: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
    region_name: Optional[str] = None,
    client=None,
    runtime_session_id: Optional[str] = None,
) -> SemanticAnalysisResult:
    """Analyze one file through an existing AgentCore Harness.

    Args:
        harness_arn: ARN of an existing harness (defaults to the
            SMS_AGENTCORE_HARNESS_ARN env var). Never created here.
        system_prompt: Per-invocation instruction override. SMS's own
            prompt keeps product behavior explicit from the caller side.
        client: Optional pre-built bedrock-agentcore client (tests).
        runtime_session_id: Optional session id; must be >= 33 chars.
            Defaults to a fresh uuid4() (stateless analysis, same as the
            SMS pipeline's per-file call pattern).
    """
    if not harness_arn:
        harness_arn = os.getenv(HARNESS_ARN_ENV)
    if not harness_arn:
        raise AgentCoreError(
            f"{HARNESS_ARN_ENV} must be set to call an AgentCore harness"
        )

    client = client or _default_client(region_name)
    session_id = runtime_session_id or uuid.uuid4().hex + "sms"
    if len(session_id) < 33:
        raise AgentCoreError("runtime_session_id must be at least 33 characters")

    response = client.invoke_harness(
        harnessArn=harness_arn,
        runtimeSessionId=session_id,
        messages=build_messages(file_key, content, metadata),
        systemPrompt=[{"text": system_prompt}],
    )
    text = _stream_to_text(response.get("stream"))
    return _parse_result_text(text, file_key)