import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from sms_agent.s3_inventory import S3InventoryCollector

def test_s3_inventory_empty_bucket():
    # Mock boto3 client
    mock_s3 = MagicMock()
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    # Simulate empty bucket (no Contents)
    mock_paginator.paginate.return_value = [{'IsTruncated': False}]
    
    collector = S3InventoryCollector("test-bucket", s3_client=mock_s3)
    result = collector.collect()
    
    assert len(result) == 0

def test_s3_inventory_mapping():
    mock_s3 = MagicMock()
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    
    dt = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
    
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {
                'Key': 'demo/file1.txt',
                'LastModified': dt,
                'Size': 1234,
                'ETag': '"fake-etag-1"'
            },
            {
                'Key': 'demo/file2.txt',
                'LastModified': dt,
                'Size': 5678,
                'ETag': '"fake-etag-2"'
            },
            {
                'Key': 'demo/duplicate.txt',
                'LastModified': dt,
                'Size': 1234,
                'ETag': '"fake-etag-1"' # Duplicate ETag
            }
        ]
    }]
    
    collector = S3InventoryCollector("test-bucket", s3_client=mock_s3)
    result = collector.collect()
    
    assert len(result) == 3
    
    # Check mapping
    assert result[0].key == "demo/file1.txt"
    assert result[0].filename == "file1.txt"
    assert result[0].extension == ".txt"
    assert result[0].size_bytes == 1234
    assert result[0].created_at == dt
    assert result[0].etag == "fake-etag-1"
    assert result[0].s3_uri == "s3://test-bucket/demo/file1.txt"
    assert result[0].bucket == "test-bucket"
    
    # Check duplicates via ETag
    assert result[0].is_duplicate is True
    assert result[1].is_duplicate is False
    assert result[2].is_duplicate is True
    
    # Check that directory placeholders are ignored
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {
                'Key': 'demo/',
                'LastModified': dt,
                'Size': 0,
                'ETag': '"dir-etag"'
            }
        ]
    }]
    
    collector2 = S3InventoryCollector("test-bucket", s3_client=mock_s3)
    result2 = collector2.collect()
    assert len(result2) == 0
