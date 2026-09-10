import pytest
from unittest.mock import patch, MagicMock
from sms_agent.pipeline import SMSPipeline
from sms_agent.models import FileMetadata, RetrievedContent, SemanticMemoryRecord

@pytest.fixture
def mock_pipeline():
    with patch('boto3.client'):
        pipeline = SMSPipeline("test-bucket")
        
        # Setup fake inventory and reader
        mock_meta = FileMetadata(
            bucket="test-bucket",
            key="test.txt",
            size_bytes=100,
            created_at="2026-09-01T00:00:00Z",
            last_modified_at="2026-09-01T00:00:00Z",
            etag="123"
        )
        pipeline.inventory.collect = MagicMock(return_value=[mock_meta])
        
        mock_content = RetrievedContent(
            bucket="test-bucket", 
            key="test.txt", 
            content="Hello World", 
            content_type="text/plain",
            size_bytes=11
        )
        pipeline.reader.get_text = MagicMock(return_value=mock_content)
        return pipeline

def test_inference_disabled_path(mock_pipeline):
    # Call with skip_inference=True
    result = mock_pipeline.process_object("test.txt", skip_inference=True)
    
    assert result["status"] == "READY_FOR_INFERENCE"
    assert result["key"] == "test.txt"
    assert result["content_size"] == 11
    assert "partial_importance_score" in result

@patch('sms_agent.agent.SMSAgent.analyze_file')
def test_pipeline_behavior_when_inference_unavailable(mock_analyze, mock_pipeline):
    # Simulate Bedrock failure
    mock_analyze.side_effect = ValueError("Bedrock/Strands inference failed: Operation not allowed")
    
    with pytest.raises(ValueError, match="Bedrock/Strands inference failed: Operation not allowed"):
        mock_pipeline.process_object("test.txt", skip_inference=False)
        
    # Verify no memory store persistence occurred because the pipeline aborted early
    assert mock_pipeline.memory_store is None
