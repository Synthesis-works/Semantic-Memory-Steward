import pytest
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

