"""AgentCore Harness adapter tests (hermetic: mocked bedrock-agentcore client).

Covers: stream unwrapping, streamed error surfacing, JSON parsing into
SemanticAnalysisResult, InvokeHarness call construction, missing-ARN
guard, and the SMSAgent.analyze_file provider wiring for
SMS_LLM_PROVIDER=agentcore. No AWS resources are created or invoked.
"""
import os

import pytest
from unittest.mock import patch

import sms_agent.agentcore as agentcore
from sms_agent.agent import SMSAgent
from sms_agent.agentcore import (
    AgentCoreError,
    analyze_file_with_harness,
    _stream_to_text,
)
from sms_agent.models import SemanticAnalysisResult

HARNESS = "arn:aws:bedrock-agentcore:us-east-1:527557823928:harness/sms-analyzer"
VALID_JSON = (
    '{"category":"financial","sensitivity":"confidential",'
    '"importance_score":0.82,"confidence":0.9,'
    '"reasoning":"Q3 draft with revenue figures","recommended_action":"review"}'
)


def _text_delta(chunk):
    return {"contentBlockDelta": {"delta": {"text": chunk}}}


def test_stream_to_text_concatenates_deltas():
    stream = [
        {"messageStart": {"role": "assistant"}},
        _text_delta('{"category":'),
        _text_delta('"financial","sensitivity":'),
        _text_delta('"confidential"}'),
        {"contentBlockStop": {}},
        {"metadata": {"tokenUsage": {"inputTokens": 10}}},
    ]
    assert _stream_to_text(stream) == '{"category":"financial","sensitivity":"confidential"}'


def test_stream_to_text_ignores_tool_blocks():
    stream = [
        _text_delta("done"),
        {"contentBlockDelta": {"delta": {"toolUse": {"name": "x"}}}},
    ]
    assert _stream_to_text(stream) == "done"


@pytest.mark.parametrize("error_key", ["validationException", "internalServerException", "runtimeClientError"])
def test_stream_to_text_surfaces_errors(error_key):
    stream = [_text_delta("partial"), {error_key: {"message": "boom"}}]
    with pytest.raises(AgentCoreError) as exc:
        _stream_to_text(stream)
    assert error_key in str(exc.value)


def test_stream_to_text_empty_stream_raises():
    with pytest.raises(AgentCoreError):
        _stream_to_text([])


def test_parse_result_text_roundtrip_and_key_override():
    result = agentcore._parse_result_text(VALID_JSON, "demo/q3.txt")
    assert isinstance(result, SemanticAnalysisResult)
    assert result.key == "demo/q3.txt"
    assert result.recommended_action == "review"
    assert result.sensitivity == "confidential"


def test_parse_normalizes_action_case_and_fences():
    text = "```json\n" + VALID_JSON.replace('"review"', '"REVIEW"') + "\n```"
    result = agentcore._parse_result_text(text, "demo/q3.txt")
    assert result.recommended_action == "review"


def test_analyze_invokes_harness_and_parses():
    calls = {}

    class FakeClient:
        def invoke_harness(self, **kwargs):
            calls.update(kwargs)
            return {"stream": [
                {"messageStart": {"role": "assistant"}},
                _text_delta(VALID_JSON),
                {"messageStop": {"stopReason": "end_turn"}},
            ]}

    result = analyze_file_with_harness(
        "demo/q3.txt",
        "Q3 content",
        {"owner": "finance"},
        harness_arn=HARNESS,
        system_prompt="classify",
        client=FakeClient(),
    )
    assert result.key == "demo/q3.txt"
    assert calls["harnessArn"] == HARNESS
    assert calls["systemPrompt"] == [{"text": "classify"}]
    assert len(calls["runtimeSessionId"]) >= 33
    message_text = calls["messages"][0]["content"][0]["text"]
    assert "demo/q3.txt" in message_text and "Q3 content" in message_text


def test_analyze_requires_harness_arn():
    os.environ.pop("SMS_AGENTCORE_HARNESS_ARN", None)
    with pytest.raises(AgentCoreError) as exc:
        analyze_file_with_harness("demo/q3.txt", "content", client=object())
    assert "SMS_AGENTCORE_HARNESS_ARN" in str(exc.value)


def test_analyze_rejects_short_session_id():
    with pytest.raises(AgentCoreError) as exc:
        analyze_file_with_harness(
            "demo/q3.txt",
            "content",
            harness_arn=HARNESS,
            runtime_session_id="short",
            client=object(),
        )
    assert "33 characters" in str(exc.value)


def test_agent_provider_wiring_uses_harness():
    os.environ["SMS_LLM_PROVIDER"] = "agentcore"
    os.environ["SMS_AGENTCORE_HARNESS_ARN"] = HARNESS

    class FakeClient:
        def invoke_harness(self, **kwargs):
            return {"stream": [_text_delta(VALID_JSON)]}

    with patch("sms_agent.agentcore._default_client", return_value=FakeClient()):
        agent = SMSAgent()
        result = agent.analyze_file("demo/q3.txt", "Q3 content")

    assert result.key == "demo/q3.txt"
    assert result.recommended_action == "review"


def test_agent_provider_requires_arn_env():
    os.environ["SMS_LLM_PROVIDER"] = "agentcore"
    os.environ.pop("SMS_AGENTCORE_HARNESS_ARN", None)
    with pytest.raises(ValueError) as exc:
        SMSAgent().analyze_file("demo/q3.txt", "content")
    assert "SMS_AGENTCORE_HARNESS_ARN" in str(exc.value)