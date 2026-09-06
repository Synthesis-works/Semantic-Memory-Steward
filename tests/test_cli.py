import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from sms_agent.__main__ import main, run_analyze_s3
from sms_agent.models import FileMetadata

@patch.dict(os.environ, {"SMS_S3_BUCKET": ""})
def test_analyze_s3_missing_bucket(capsys):
    """Test that analyze-s3 fails cleanly if the bucket is missing."""
    with patch.object(sys, 'argv', ['python -m sms_agent', 'analyze-s3']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
        captured = capsys.readouterr()
        assert "SMS_S3_BUCKET environment variable must be set" in captured.out

@patch.dict(os.environ, {"SMS_S3_BUCKET": "test-bucket"})
@patch("sms_agent.__main__.S3InventoryCollector")
@patch("sms_agent.__main__.SMSPipeline")
def test_analyze_s3_success(mock_pipeline_cls, mock_inventory_cls, capsys):
    """Test that analyze-s3 processes objects and displays output correctly."""
    mock_inventory = MagicMock()
    # Mock two files: one normal, one duplicate
    import datetime
    dt = datetime.datetime.now(datetime.timezone.utc)
    mock_inventory.collect.return_value = [
        FileMetadata(key="doc1.txt", extension=".txt", size_bytes=100, created_at=dt, bucket="test-bucket"),
        FileMetadata(key="doc2.txt", extension=".txt", size_bytes=100, created_at=dt, bucket="test-bucket")
    ]
    mock_inventory_cls.return_value = mock_inventory
    
    mock_pipeline = MagicMock()
    # First file is a KEEP
    def mock_process_object(key, skip_inference, execute_action):
        assert skip_inference is False
        assert execute_action is False
        if key == "doc1.txt":
            return {
                "analysis": {"category": "Financial Reporting", "sensitivity": "confidential"},
                "importance": {"score": 0.85},
                "decision": {"action": "KEEP", "requires_human_approval": False},
                "action_request": {"requested_action": "NONE"}
            }
        elif key == "doc2.txt":
            # Second file is a QUARANTINE duplicate
            return {
                "analysis": {"category": "Technical", "sensitivity": "public"},
                "importance": {"score": 0.25},
                "relationships": [{"relationship_type": "duplicate_candidate", "related_object": "doc1.txt", "confidence": 0.99}],
                "decision": {"action": "REVIEW", "requires_human_approval": True},
                "action_request": {"requested_action": "REVIEW"}
            }
        
    mock_pipeline.process_object.side_effect = mock_process_object
    mock_pipeline_cls.return_value = mock_pipeline
    
    with patch.object(sys, 'argv', ['python -m sms_agent', 'analyze-s3']):
        main()
        
    captured = capsys.readouterr()
    out = captured.out
    
    assert "Objects discovered: 2" in out
    
    # Check doc1 output
    assert "doc1.txt" in out
    assert "Category: Financial Reporting" in out
    assert "Sensitivity: Confidential" in out
    assert "Importance: 0.85" in out
    assert "Recommendation: KEEP" in out
    assert "Policy: NONE" in out
    
    # Check doc2 output
    assert "doc2.txt" in out
    assert "Relationship: Duplicate Candidate" in out
    assert "Related to: doc1.txt" in out
    assert "Recommendation: REVIEW" in out
    assert "Policy: REVIEW" in out
    assert "Approval: Required" in out
    assert "Action: Not Executed" in out

@patch.dict(os.environ, {"SMS_S3_BUCKET": "test-bucket"})
@patch("sms_agent.__main__.S3InventoryCollector")
@patch("sms_agent.__main__.SMSPipeline")
def test_analyze_s3_error_handling(mock_pipeline_cls, mock_inventory_cls, capsys):
    """Test that analyze-s3 handles individual object failures safely without crashing."""
    mock_inventory = MagicMock()
    import datetime
    dt = datetime.datetime.now(datetime.timezone.utc)
    mock_inventory.collect.return_value = [
        FileMetadata(key="doc1.txt", extension=".txt", size_bytes=100, created_at=dt, bucket="test-bucket"),
        FileMetadata(key="doc2.txt", extension=".txt", size_bytes=100, created_at=dt, bucket="test-bucket")
    ]
    mock_inventory_cls.return_value = mock_inventory
    
    mock_pipeline = MagicMock()
    def mock_process_object(key, skip_inference, execute_action):
        if key == "doc1.txt":
            raise ValueError("Unsupported content type")
        return {
            "decision": {"action": "KEEP"}
        }
    
    mock_pipeline.process_object.side_effect = mock_process_object
    mock_pipeline_cls.return_value = mock_pipeline
    
    with patch.object(sys, 'argv', ['python -m sms_agent', 'analyze-s3']):
        main()
        
    captured = capsys.readouterr()
    out = captured.out
    
    assert "doc1.txt" in out
    assert "ANALYSIS FAILED" in out
    assert "Reason: Unsupported content type" in out
    
    assert "doc2.txt" in out
    assert "Recommendation: KEEP" in out
