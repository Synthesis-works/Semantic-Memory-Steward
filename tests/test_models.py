import pytest
from pydantic import ValidationError
from sms_agent.models import SemanticAnalysisResult

def test_valid_analysis_result():
    data = {
        "key": "test.txt",
        "category": "technical",
        "sensitivity": "internal",
        "importance_score": 0.8,
        "confidence": 0.9,
        "reasoning": "Standard technical document.",
        "recommended_action": "retain"
    }
    result = SemanticAnalysisResult(**data)
    assert result.key == "test.txt"
    assert result.importance_score == 0.8
    assert result.sensitivity == "internal"

def test_invalid_score():
    data = {
        "key": "test.txt",
        "category": "technical",
        "sensitivity": "internal",
        "importance_score": 1.5, # invalid, > 1.0
        "confidence": 0.9,
        "reasoning": "...",
        "recommended_action": "retain"
    }
    with pytest.raises(ValidationError):
        SemanticAnalysisResult(**data)

def test_invalid_sensitivity():
    data = {
        "key": "test.txt",
        "category": "technical",
        "sensitivity": "top-secret", # not in Literal
        "importance_score": 0.5,
        "confidence": 0.9,
        "reasoning": "...",
        "recommended_action": "retain"
    }
    with pytest.raises(ValidationError):
        SemanticAnalysisResult(**data)
