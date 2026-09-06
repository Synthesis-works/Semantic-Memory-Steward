import os
from datetime import datetime, timezone
from typing import List, Dict
from pathlib import Path
import boto3

from .models import FileMetadata

class S3InventoryCollector:
    """Collects metadata from a real Amazon S3 bucket."""

    def __init__(self, bucket_name: str, prefix: str = "", s3_client=None):
        self.bucket_name = bucket_name
        self.prefix = prefix
        # Allow passing an injected client for testing, otherwise create one
        self.s3 = s3_client or boto3.client('s3')

    def collect(self) -> List[FileMetadata]:
        inventory: List[FileMetadata] = []
        paginator = self.s3.get_paginator('list_objects_v2')
        
        etag_registry: Dict[str, bool] = {}

        try:
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip directory placeholders if present
                    if key.endswith('/'):
                        continue
                        
                    filename = os.path.basename(key)
                    _, ext = os.path.splitext(filename)
                    size = obj['Size']
                    last_modified = obj['LastModified']  # datetime object from boto3
                    etag = obj.get('ETag', '').strip('"') # S3 ETags are often quoted
                    
                    s3_uri = f"s3://{self.bucket_name}/{key}"
                    
                    meta = FileMetadata(
                        key=key,
                        filename=filename,
                        extension=ext.lower() if ext else "",
                        size_bytes=size,
                        created_at=last_modified,      # S3 doesn't have true created_at, fallback to LastModified
                        last_modified_at=last_modified,
                        etag=etag,
                        s3_uri=s3_uri,
                        bucket=self.bucket_name,
                        is_duplicate=False
                    )
                    inventory.append(meta)
                    
                    if etag:
                        if etag in etag_registry:
                            etag_registry[etag] = True
                        else:
                            etag_registry[etag] = False
                            
        except Exception as e:
            print(f"Error reading from S3 bucket {self.bucket_name}: {e}")
            
        # Second pass to mark duplicates based on ETag (which is often a proxy for content hash)
        for meta in inventory:
            if meta.etag and etag_registry.get(meta.etag, False):
                meta.is_duplicate = True
                
        return inventory
