import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from sms_agent.pipeline import SMSPipeline
from sms_agent.models import SemanticAnalysisResult, PolicyDecision

def test_pipeline_skip_inference():
    # Mock boto3
    mock_s3 = MagicMock()
    
    # Mock inventory list_objects_v2
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    dt = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {
                'Key': 'demo/doc.txt',
                'LastModified': dt,
                'Size': 12,
                'ETag': '"etag"'
            },
            {
                'Key': 'demo/doc-copy.txt',
                'LastModified': dt,
                'Size': 12,
                'ETag': '"etag"'
            }
        ]
    }]
    
    # Mock content reader get_object
    mock_body = MagicMock()
    mock_body.read.return_value = b"Hello, test!"
    mock_s3.get_object.return_value = {
        'Body': mock_body,
        'ContentType': 'text/plain'
    }
    
    pipeline = SMSPipeline(bucket_name="test-bucket", s3_client=mock_s3)
    
    result = pipeline.process_object("demo/doc.txt", skip_inference=True)
    
    assert result["status"] == "READY_FOR_INFERENCE"
    assert result["bucket"] == "test-bucket"
    assert result["key"] == "demo/doc.txt"
    assert result["content_size"] == 12
    assert result["content_preview"] == "Hello, test!"
    assert result["metadata"]["size_bytes"] == 12
    assert "relationships" in result
    assert isinstance(result["relationships"], list)
    assert len(result["relationships"]) == 1
    assert result["relationships"][0]["relationship_type"] == "DUPLICATE_CANDIDATE"
    assert "partial_importance_score" in result
    assert result["partial_importance_score"]["score"] >= 0.0

def test_pipeline_not_found():
    mock_s3 = MagicMock()
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{'IsTruncated': False}] # Empty bucket
    
    pipeline = SMSPipeline(bucket_name="test-bucket", s3_client=mock_s3)
    
    with pytest.raises(ValueError, match="Object metadata not found"):
        pipeline.process_object("nonexistent.txt")
