import pytest
from unittest.mock import patch, MagicMock
from sms_agent.agent import SMSAgent
from sms_agent.models import SemanticAnalysisResult
import os

@pytest.fixture
def agent():
    with patch.dict(os.environ, {"SMS_LLM_PROVIDER": "bedrock", "SMS_BEDROCK_MODEL_ID": "amazon.nova-micro-v1:0"}):
        return SMSAgent()

def test_strands_agent_construction(agent):
    assert agent.model_id == "amazon.nova-micro-v1:0"

    assert "Semantic Memory Steward" in agent.agent.system_prompt

@patch('strands.Agent.structured_output')
def test_structured_semantic_result_handling(mock_structured, agent):
    # Mock structured output returning a valid Pydantic model
    mock_result = SemanticAnalysisResult(
        key="test.txt",
        category="hr",
        sensitivity="internal",
        importance_score=0.8,
        confidence=0.9,
        reasoning="Test reasoning",
        recommended_action="retain"
    )
    mock_structured.return_value = mock_result
    
    result = agent.analyze_file("test.txt", "some content", {"size": 100})
    
    assert result.category == "hr"
    assert result.sensitivity == "internal"
    assert result.importance_score == 0.8
    assert result.key == "test.txt"
    mock_structured.assert_called_once()

@patch('strands.Agent.structured_output')
def test_invalid_model_output(mock_structured, agent):
    # Simulate an error raised by Strands during structured output parsing
    mock_structured.side_effect = ValueError("Strands parsing error: Invalid JSON")
    
    with pytest.raises(ValueError, match="Bedrock/Strands inference failed: Strands parsing error: Invalid JSON"):
        agent.analyze_file("test.txt", "some content", {})

@patch('strands.Agent.structured_output')
def test_bedrock_inference_error_propagation(mock_structured, agent):
    # Simulate a Bedrock AccessDenied or Operation Not Allowed exception
    mock_structured.side_effect = Exception("ValidationException: Operation not allowed")
    
    with pytest.raises(ValueError, match="Bedrock/Strands inference failed: ValidationException: Operation not allowed"):
        agent.analyze_file("test.txt", "some content", {})
