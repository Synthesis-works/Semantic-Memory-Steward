import pytest
import os
import json
from unittest.mock import patch, MagicMock
from sms_agent.agent import SMSAgent
from sms_agent.models import SemanticAnalysisResult

@patch("sms_agent.agent.SMSAgent._call_gemini")
def test_agent_parsing_mocked(mock_call):
    """Test that the agent correctly parses a valid JSON response from the underlying model."""
    mock_call.return_value = json.dumps({
        "key": "mock-file.txt",
        "category": "technical",
        "sensitivity": "internal",
        "importance_score": 0.5,
        "confidence": 0.9,
        "reasoning": "Mocked local response",
        "recommended_action": "retain"
    })

    with patch.dict(os.environ, {"SMS_LLM_PROVIDER": "gemini"}):
        agent = SMSAgent()
        result = agent.analyze_file("mock-file.txt", "Some content", {"meta": "data"})

    assert isinstance(result, SemanticAnalysisResult)
    assert result.key == "mock-file.txt"
    assert result.recommended_action == "retain"
    assert result.confidence == 0.9

@patch("sms_agent.agent.SMSAgent._call_gemini")
def test_agent_invalid_json(mock_call):
    """Test handling of invalid JSON from the model."""
    mock_call.return_value = "This is not JSON"

    with patch.dict(os.environ, {"SMS_LLM_PROVIDER": "gemini"}):
        agent = SMSAgent()

        with pytest.raises(ValueError, match="Failed to parse agent response"):
            agent.analyze_file("mock-file.txt", "Some content", {})


def _mock_gemini_response(mock_urlopen):
    """Wire a valid Gemini-shaped response into the urlopen mock."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "key": "test.txt",
                        "category": "technical",
                        "sensitivity": "public",
                        "importance_score": 0.2,
                        "confidence": 0.85,
                        "reasoning": "Mocked Gemini Response",
                        "recommended_action": "retain"
                    })
                }]
            }
        }]
    }).encode('utf-8')
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response


@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "gemini", "SMS_EXTERNAL_API_KEY": "AQ.test-new-format-key"})
@patch("urllib.request.urlopen")
def test_gemini_new_format_key_uses_query_param(mock_urlopen):
    """AQ.*-format AI Studio keys must also use the ?key= transport.

    Regression: these were briefly misrouted to Bearer (which Google
    rejects for API keys); verified live that ?key= is correct.
    """
    _mock_gemini_response(mock_urlopen)
    agent = SMSAgent()
    agent.analyze_file("test.txt", "content")
    req = mock_urlopen.call_args[0][0]
    assert "key=AQ.test-new-format-key" in req.full_url
    assert req.get_header("Authorization") is None


@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "gemini", "SMS_EXTERNAL_API_KEY": "AIza-test-api-key"})
@patch("urllib.request.urlopen")
def test_gemini_api_key_uses_query_param(mock_urlopen):
    """Classic AIza.* API keys keep using the ?key= transport."""
    _mock_gemini_response(mock_urlopen)
    agent = SMSAgent()
    agent.analyze_file("test.txt", "content")
    req = mock_urlopen.call_args[0][0]
    assert "key=AIza-test-api-key" in req.full_url
    assert req.get_header("Authorization") is None

def test_strands_agent_import():
    """Verify that the real strands Agent can be imported."""
    from strands import Agent
    assert Agent is not None

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "gemini", "SMS_EXTERNAL_API_KEY": "test-key"})
@patch("urllib.request.urlopen")
def test_agent_gemini_provider(mock_urlopen):
    """Test that the Gemini external provider handles the call properly."""
    # Mock urllib response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "key": "test.txt",
                        "category": "financial",
                        "sensitivity": "restricted",
                        "importance_score": 0.9,
                        "confidence": 0.95,
                        "reasoning": "Mocked Gemini Response",
                        "recommended_action": "review"
                    })
                }]
            }
        }]
    }).encode('utf-8')
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    agent = SMSAgent()
    result = agent.analyze_file("test.txt", "content")

    assert isinstance(result, SemanticAnalysisResult)
    assert result.key == "test.txt"
    assert result.category == "financial"
    assert result.recommended_action == "review"
    assert mock_urlopen.called

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "groq", "SMS_EXTERNAL_API_KEY": "test-key"})
@patch("urllib.request.urlopen")
def test_agent_groq_provider(mock_urlopen):
    """Test that the Groq external provider handles the call properly."""
    # Mock urllib response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "key": "test.txt",
                    "category": "technical",
                    "sensitivity": "public",
                    "importance_score": 0.2,
                    "confidence": 0.85,
                    "reasoning": "Mocked Groq Response",
                    "recommended_action": "retain"
                })
            }
        }]
    }).encode('utf-8')
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    agent = SMSAgent()
    result = agent.analyze_file("test.txt", "content")

    assert isinstance(result, SemanticAnalysisResult)
    assert result.key == "test.txt"
    assert result.category == "technical"
    assert result.recommended_action == "retain"
    assert mock_urlopen.called

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "groq", "SMS_EXTERNAL_API_KEY": ""})
def test_agent_groq_missing_key():
    """Test that missing API key for Groq raises ValueError."""
    agent = SMSAgent()
    with pytest.raises(ValueError, match="SMS_EXTERNAL_API_KEY is not set"):
        agent.analyze_file("test.txt", "content")

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "mistral", "SMS_EXTERNAL_API_KEY": "test-key"})
@patch("urllib.request.urlopen")
def test_agent_mistral_provider(mock_urlopen):
    """Test that the Mistral external provider handles the call properly."""
    # Mock urllib response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "key": "test.txt",
                    "category": "technical",
                    "sensitivity": "public",
                    "importance_score": 0.2,
                    "confidence": 0.85,
                    "reasoning": "Mocked Mistral Response",
                    "recommended_action": "retain"
                })
            }
        }]
    }).encode('utf-8')
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    agent = SMSAgent()
    result = agent.analyze_file("test.txt", "content")

    assert isinstance(result, SemanticAnalysisResult)
    assert result.key == "test.txt"
    assert result.category == "technical"
    assert result.recommended_action == "retain"
    assert mock_urlopen.called

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "mistral", "SMS_EXTERNAL_API_KEY": ""})
def test_agent_mistral_missing_key():
    """Test that missing API key for Mistral raises ValueError."""
    agent = SMSAgent()
    with pytest.raises(ValueError, match="SMS_EXTERNAL_API_KEY is not set"):
        agent.analyze_file("test.txt", "content")

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "nvidia", "SMS_EXTERNAL_API_KEY": "test-key"})
@patch("urllib.request.urlopen")
def test_agent_nvidia_provider(mock_urlopen):
    """Test that the NVIDIA external provider handles the call properly."""
    # Mock urllib response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "key": "test.txt",
                    "category": "technical",
                    "sensitivity": "public",
                    "importance_score": 0.2,
                    "confidence": 0.85,
                    "reasoning": "Mocked NVIDIA Response",
                    "recommended_action": "retain"
                })
            }
        }]
    }).encode('utf-8')
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    agent = SMSAgent()
    result = agent.analyze_file("test.txt", "content")

    assert isinstance(result, SemanticAnalysisResult)
    assert result.key == "test.txt"
    assert result.category == "technical"
    assert result.recommended_action == "retain"
    assert mock_urlopen.called

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "nvidia", "SMS_EXTERNAL_API_KEY": ""})
def test_agent_nvidia_missing_key():
    """Test that missing API key for NVIDIA raises ValueError."""
    agent = SMSAgent()
    with pytest.raises(ValueError, match="SMS_EXTERNAL_API_KEY is not set"):
        agent.analyze_file("test.txt", "content")

@patch.dict(os.environ, {"SMS_LLM_PROVIDER": "gemini", "SMS_EXTERNAL_API_KEY": ""})
def test_agent_gemini_missing_key():
    """Test that missing API key raises ValueError."""
    agent = SMSAgent()
    with pytest.raises(ValueError, match="SMS_EXTERNAL_API_KEY is not set"):
        agent.analyze_file("test.txt", "content")


def test_translate_strands_tool_use_event():
    """A real Strands tool-use event becomes a trace event with the tool name."""
    from sms_agent.agent import translate_strands_event
    from sms_agent.trace import TraceRecorder
    rec = TraceRecorder()
    translate_strands_event(rec, "demo/a.txt",
                            current_tool_use={"name": "s3_read"})
    assert len(rec.events) == 1
    assert rec.events[0].stage == "AI_ANALYSIS"
    assert "s3_read" in rec.events[0].title


def test_translate_strands_ignores_non_tool_events():
    """Model output/reasoning/lifecycle noise must never enter the trace."""
    from sms_agent.agent import translate_strands_event
    from sms_agent.trace import TraceRecorder
    rec = TraceRecorder()
    translate_strands_event(rec, "demo/a.txt", data="token stream...")
    translate_strands_event(rec, "demo/a.txt", reasoningText="private thought")
    translate_strands_event(rec, "demo/a.txt", current_tool_use={})
    translate_strands_event(None, "demo/a.txt",
                            current_tool_use={"name": "x"})
    assert rec.events == []


def test_translate_strands_never_raises():
    """Tracing must not break inference, whatever the SDK sends."""
    from sms_agent.agent import translate_strands_event
    from sms_agent.trace import TraceRecorder
    rec = TraceRecorder()
    translate_strands_event(rec, "demo/a.txt", event=None,
                            current_tool_use="not-a-dict")
    assert rec.events == []

