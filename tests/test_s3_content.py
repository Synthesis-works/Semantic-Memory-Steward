import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from sms_agent.models import FileMetadata
from sms_agent.s3_content import S3ContentReader

@pytest.fixture
def dummy_metadata():
    return FileMetadata(
        key="demo/file.txt",
        filename="file.txt",
        extension=".txt",
        size_bytes=100,
        created_at=datetime.now(timezone.utc),
        bucket="test-bucket"
    )

def test_s3_content_success(dummy_metadata):
    mock_s3 = MagicMock()
    # Mock the get_object response
    mock_body = MagicMock()
    mock_body.read.return_value = b"Hello, world!"
    mock_s3.get_object.return_value = {
        'Body': mock_body,
        'ContentType': 'text/plain'
    }
    
    reader = S3ContentReader(s3_client=mock_s3)
    content = reader.get_text(dummy_metadata)
    
    assert content.content == "Hello, world!"
    assert content.size_bytes == 13
    assert content.bucket == "test-bucket"
    assert content.key == "demo/file.txt"
    mock_s3.get_object.assert_called_once_with(Bucket="test-bucket", Key="demo/file.txt")

def test_s3_content_unsupported_type(dummy_metadata):
    dummy_metadata.extension = ".docx"

    reader = S3ContentReader()
    with pytest.raises(ValueError, match="Unsupported content type"):
        reader.get_text(dummy_metadata)

def test_s3_content_too_large(dummy_metadata):
    dummy_metadata.size_bytes = 200_000
    
    reader = S3ContentReader(max_bytes=100_000)
    with pytest.raises(ValueError, match="exceeds maximum allowed size"):
        reader.get_text(dummy_metadata)

def test_s3_content_empty_object(dummy_metadata):
    dummy_metadata.size_bytes = 0
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.read.return_value = b""
    mock_s3.get_object.return_value = {
        'Body': mock_body,
        'ContentType': 'text/plain'
    }
    
    reader = S3ContentReader(s3_client=mock_s3)
    content = reader.get_text(dummy_metadata)
    assert content.content == ""
    assert content.size_bytes == 0

def test_s3_content_s3_error(dummy_metadata):
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("Access Denied")
    
    reader = S3ContentReader(s3_client=mock_s3)
    with pytest.raises(RuntimeError, match="Failed to retrieve"):
        reader.get_text(dummy_metadata)
