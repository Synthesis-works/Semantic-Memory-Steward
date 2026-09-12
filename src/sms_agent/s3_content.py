import html as html_module
import io
import os
from html.parser import HTMLParser
from typing import Optional

import boto3

from .models import FileMetadata, RetrievedContent


class _VisibleTextExtractor(HTMLParser):
    """Collect visible text, dropping tags plus script/style content."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.parts.append(text)

    def text(self):
        return html_module.unescape("\n".join(self.parts))


def extract_html_text(raw: str) -> str:
    parser = _VisibleTextExtractor()
    parser.feed(raw)
    return parser.text()


def extract_pdf_text(body: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(body))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(part for part in (p.strip() for p in pages) if part)
    if not text:
        raise ValueError("PDF contains no extractable text.")
    return text


def extract_xlsx_text(body: bytes) -> str:
    from openpyxl import load_workbook
    workbook = load_workbook(io.BytesIO(body), read_only=True,
                             data_only=True)
    blocks = []
    for sheet in workbook.worksheets:
        blocks.append(f"Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(value) for value in row if value is not None]
            if cells:
                blocks.append(" | ".join(cells))
    text = "\n".join(blocks).strip()
    if not text:
        raise ValueError("Workbook contains no extractable cell values.")
    return text


class S3ContentReader:
    """Reads actual text content from Amazon S3 for semantic analysis."""

    ALLOWED_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.html',
                          '.pdf', '.xlsx'}
    
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
            if ext == ".pdf":
                text_content = extract_pdf_text(body)
            elif ext == ".xlsx":
                text_content = extract_xlsx_text(body)
            else:
                # Decode carefully (handle UTF-16 BOM if present)
                if body.startswith(b'\xff\xfe') or body.startswith(b'\xfe\xff'):
                    text_content = body.decode('utf-16', errors='replace')
                else:
                    text_content = body.decode('utf-8', errors='replace')
                if ext == ".html":
                    text_content = extract_html_text(text_content)
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
