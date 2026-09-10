import pytest
from unittest.mock import MagicMock
from sms_agent.comprehend import ComprehendAnalyzer

def test_analyze_empty_text():
    analyzer = ComprehendAnalyzer(client=MagicMock())
    result = analyzer.analyze_text("")
    assert result["entities"] == []
    assert result["pii_entities"] == []
    assert result["errors"] == []

def test_analyze_success():
    mock_client = MagicMock()
    mock_client.detect_entities.return_value = {
        'Entities': [
            {'Type': 'ORGANIZATION', 'Score': 0.99, 'Text': 'Amazon', 'BeginOffset': 0, 'EndOffset': 6}
        ]
    }
    mock_client.detect_pii_entities.return_value = {
        'PiiEntities': [
            {'Type': 'EMAIL', 'Score': 0.95, 'BeginOffset': 10, 'EndOffset': 25}
        ]
    }
    
    analyzer = ComprehendAnalyzer(client=mock_client)
    result = analyzer.analyze_text("Some random text with email@example.com")
    
    assert len(result["entities"]) == 1
    assert result["entities"][0]["type"] == "ORGANIZATION"
    assert result["entities"][0]["score"] == 0.99
    assert "Text" not in result["entities"][0]  # Ensure raw text is omitted
    
    assert len(result["pii_entities"]) == 1
    assert result["pii_entities"][0]["type"] == "EMAIL"
    assert result["pii_entities"][0]["score"] == 0.95
    assert "Text" not in result["pii_entities"][0] # Just to be sure

def test_analyze_aws_exception():
    mock_client = MagicMock()
    mock_client.detect_entities.side_effect = Exception("AWS Error")
    mock_client.detect_pii_entities.return_value = {'PiiEntities': []}
    
    analyzer = ComprehendAnalyzer(client=mock_client)
    result = analyzer.analyze_text("Hello")
    
    assert len(result["errors"]) == 1
    assert "AWS Error" in result["errors"][0]
    assert result["entities"] == []
    assert result["pii_entities"] == []

def test_truncate_long_text():
    mock_client = MagicMock()
    mock_client.detect_entities.return_value = {'Entities': []}
    mock_client.detect_pii_entities.return_value = {'PiiEntities': []}
    
    analyzer = ComprehendAnalyzer(client=mock_client)
    
    # 6000 bytes string
    long_text = "A" * 6000
    analyzer.analyze_text(long_text)
    
    # Ensure it only sent 4900 bytes to AWS
    args, kwargs = mock_client.detect_entities.call_args
    assert len(kwargs['Text']) <= 4900

