"""Extraction tests: every advertised format must genuinely extract.

Fixtures in tests/fixtures/ are deterministic committed files. S3 is
mocked; no network is touched. Unsupported formats must fail honestly
(ValueError), never silently pass filenames to the LLM.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sms_agent.models import FileMetadata
from sms_agent.s3_content import S3ContentReader

FIXTURES = Path(__file__).parent / "fixtures" / "extraction"


def _reader_for(filename, payload: bytes):
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = payload
    s3.get_object.return_value = {"Body": body,
                                  "ContentType": "application/octet-stream"}
    reader = S3ContentReader(s3_client=s3)
    meta = FileMetadata(
        bucket="bucket", key=f"demo/{filename}", size_bytes=len(payload),
        created_at="2026-09-01T00:00:00Z",
        extension=Path(filename).suffix.lower())
    return reader, meta


def _read_fixture(name):
    return (FIXTURES / name).read_bytes()


def test_allowed_extensions_covers_text_plus_pdf_xlsx():
    assert S3ContentReader.ALLOWED_EXTENSIONS == {
        ".txt", ".md", ".csv", ".json", ".html", ".pdf", ".xlsx"}


def test_docx_fails_honestly():
    reader, meta = _reader_for("notes.docx", b"PK\x03\x04fake")
    with pytest.raises(ValueError, match="Unsupported"):
        reader.get_text(meta)


def test_txt_fixture_extracts():
    reader, meta = _reader_for("ops-notes.txt", _read_fixture("ops-notes.txt"))
    out = reader.get_text(meta)
    assert "Incident timeline" in out.content


def test_markdown_fixture_extracts():
    reader, meta = _reader_for("runbook.md", _read_fixture("runbook.md"))
    out = reader.get_text(meta)
    assert "Deployment Runbook" in out.content
    assert "## " in out.content  # markdown structure preserved as text


def test_csv_fixture_extracts_rows():
    reader, meta = _reader_for("inventory.csv", _read_fixture("inventory.csv"))
    out = reader.get_text(meta)
    assert "hostname,status" in out.content
    assert "web-01,healthy" in out.content


def test_json_fixture_extracts_structured_text():
    reader, meta = _reader_for("config.json", _read_fixture("config.json"))
    out = reader.get_text(meta)
    assert "quarterly-report" in out.content
    assert "Financial Reporting" in out.content


def test_html_fixture_strips_tags():
    reader, meta = _reader_for("wiki.html", _read_fixture("wiki.html"))
    out = reader.get_text(meta)
    assert "Quarterly Planning" in out.content
    assert "<h1>" not in out.content
    assert "<li>" not in out.content


def test_pdf_fixture_extracts_headings_and_body():
    reader, meta = _reader_for("incident-review.pdf",
                               _read_fixture("incident-review.pdf"))
    out = reader.get_text(meta)
    assert "Quarterly Incident Review" in out.content
    assert "Containment" in out.content
    assert "follow-up" in out.content


def test_xlsx_fixture_extracts_cells():
    reader, meta = _reader_for("budget.xlsx", _read_fixture("budget.xlsx"))
    out = reader.get_text(meta)
    assert "Revenue" in out.content
    assert "1250000" in out.content
    assert "Expenses" in out.content


def test_oversize_still_rejected():
    reader = S3ContentReader(s3_client=MagicMock(), max_bytes=10)
    meta = FileMetadata(bucket="bucket", key="demo/big.txt", size_bytes=11,
                        created_at="2026-09-01T00:00:00Z", extension=".txt")
    with pytest.raises(ValueError, match="exceeds maximum"):
        reader.get_text(meta)
