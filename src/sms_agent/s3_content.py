import os
import boto3
from typing import Optional
from .models import FileMetadata, RetrievedContent

class S3ContentReader:
    """Reads actual text content from Amazon S3 for semantic analysis."""
    
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.csv'}
    
    def __init__(self, s3_client=None, max_bytes: int = None):
        self.s3 = s3_client or boto3.client('s3')
        if max_bytes is None:
            # Default to a safe limit, configurable via environment
            env_limit = os.getenv("SMS_MAX_CONTENT_BYTES")
            self.max_bytes = int(env_limit) if env_limit else 100_000
        else:
            self.max_bytes = max_bytes

    def get_text(self, metadata: FileMetadata) -> RetrievedContent:
        """
        Retrieve text content from an S3 object based on FileMetadata.
        Raises ValueError if the object is too large or unsupported.
        """
        if not metadata.bucket:
            raise ValueError(f"FileMetadata for {metadata.key} is missing a bucket name.")
            
        ext = (metadata.extension or "").lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported content type/extension '{ext}'. Only text formats are supported in MVP.")
            
        if metadata.size_bytes > self.max_bytes:
            raise ValueError(
                f"Object {metadata.key} exceeds maximum allowed size "
                f"({metadata.size_bytes} > {self.max_bytes} bytes). Cannot safely analyze."
            )
            
        try:
            response = self.s3.get_object(Bucket=metadata.bucket, Key=metadata.key)
            body = response['Body'].read()
            # Decode carefully
            text_content = body.decode('utf-8', errors='replace')
            content_type = response.get('ContentType', 'text/plain')
            
            return RetrievedContent(
                bucket=metadata.bucket,
                key=metadata.key,
                content=text_content,
                content_type=content_type,
                size_bytes=len(body)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve {metadata.key} from S3 bucket {metadata.bucket}: {e}") from e
