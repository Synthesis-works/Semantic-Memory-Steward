import pytest
import os
import json
from unittest.mock import patch, MagicMock
from sms_agent.agent import SMSAgent
from sms_agent.models import SemanticAnalysisResult

@patch("sms_agent.agent.Agent")
def test_agent_parsing_mocked(MockAgent):
    """Test that the agent correctly parses a valid JSON response from the underlying model."""
    mock_instance = MockAgent.return_value
    mock_instance.return_value = json.dumps({
        "key": "mock-file.txt",
        "category": "technical",
        "sensitivity": "internal",
        "importance_score": 0.5,
        "confidence": 0.9,
        "reasoning": "Mocked local response",
        "recommended_action": "retain"
    })

    agent = SMSAgent()
    result = agent.analyze_file("mock-file.txt", "Some content", {"meta": "data"})

    assert isinstance(result, SemanticAnalysisResult)
    assert result.key == "mock-file.txt"
    assert result.recommended_action == "retain"
    assert result.confidence == 0.9

@patch("sms_agent.agent.Agent")
def test_agent_invalid_json(MockAgent):
    """Test handling of invalid JSON from the model."""
    mock_instance = MockAgent.return_value
    mock_instance.return_value = "This is not JSON"

    agent = SMSAgent()

    with pytest.raises(ValueError, match="Failed to parse agent response"):
        agent.analyze_file("bad.txt", "Bad content")

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

