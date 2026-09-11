"""Regression tests: KEEP decisions must persist in semantic memory.

Observed failure: REVIEW -> KEEP -> Confirm & Execute in the dashboard produced
no visible state change because a KEEP decision was recorded nowhere — the
DynamoDB record kept saying "review" and the table kept showing PENDING_REVIEW.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from sms_agent.memory import DynamoDBMemoryStore
from sms_agent.models import (
    FileMetadata,
    RetrievedContent,
    SemanticAnalysisResult,
    SemanticMemoryRecord,
)
from sms_agent.pipeline import SMSPipeline


def _record(**kwargs):
    base = dict(
        s3_uri="s3://bucket/demo/review-doc.txt",
        bucket="bucket",
        key="demo/review-doc.txt",
        etag="etag1",
        size_bytes=100,
        category="technical",
        sensitivity="confidential",
        importance_score=0.8,
        analysis_timestamp=datetime.now(timezone.utc),
        embedding_model="m",
        vector_id="v",
        recommended_action="review",
    )
    base.update(kwargs)
    return SemanticMemoryRecord(**base)


def _dict_backed_store(item=None):
    """Real DynamoDBMemoryStore wired to an in-memory dict table."""
    box = {}
    if item is not None:
        box[item["s3_uri"]] = dict(item)
    table = MagicMock()
    table.get_item.side_effect = lambda Key: (
        {"Item": box[Key["s3_uri"]]} if Key["s3_uri"] in box else {}
    )
    table.put_item.side_effect = lambda Item: box.__setitem__(Item["s3_uri"], dict(Item))
    resource = MagicMock()
    resource.Table.return_value = table
    return DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource), box


def test_human_decision_round_trips_through_save_and_get():
    store, _ = _dict_backed_store()
    store.save_record(_record(human_decision="KEEP"))
    loaded = store.get_record("s3://bucket/demo/review-doc.txt")
    assert loaded.human_decision == "KEEP"


def test_record_defaults_to_no_human_decision():
    assert _record().human_decision is None


def test_record_human_decision_persists_keep():
    store, _ = _dict_backed_store()
    store.save_record(_record())
    store.record_human_decision("s3://bucket/demo/review-doc.txt", "KEEP")
    loaded = store.get_record("s3://bucket/demo/review-doc.txt")
    assert loaded is not None
    assert loaded.human_decision == "KEEP"
    # Policy output must be untouched by the human decision marker.
    assert loaded.recommended_action == "review"


def test_record_human_decision_rejects_unknown_document():
    store, _ = _dict_backed_store()
    with pytest.raises(ValueError, match="no record"):
        store.record_human_decision("s3://bucket/demo/missing.txt", "KEEP")


def test_record_human_decision_rejects_invalid_decision():
    store, _ = _dict_backed_store()
    store.save_record(_record())
    with pytest.raises(ValueError, match="decision"):
        store.record_human_decision("s3://bucket/demo/review-doc.txt", "DELETE")


@patch("sms_agent.agent.SMSAgent.analyze_file")
def test_pipeline_rebuild_preserves_human_decision(mock_analyze):
    """Re-analysis must not clobber a recorded KEEP decision."""
    with patch("boto3.client"):
        pipeline = SMSPipeline("test-bucket")
    uri = "s3://test-bucket/test.txt"
    meta = FileMetadata(
        bucket="test-bucket",
        key="test.txt",
        size_bytes=11,
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        etag="changed-etag",
    )
    pipeline.inventory.collect = MagicMock(return_value=[meta])
    pipeline.reader.get_text = MagicMock(
        return_value=RetrievedContent(
            bucket="test-bucket", key="test.txt", content="Hello World",
            content_type="text/plain", size_bytes=11,
        )
    )
    pipeline.comprehend.analyze_text = MagicMock(
        return_value={"pii_entities": [], "entities": []}
    )
    mock_analyze.return_value = SemanticAnalysisResult(
        key="test.txt", category="technical", sensitivity="public",
        importance_score=0.2, confidence=0.9, reasoning="stale log",
        recommended_action="archive",
    )
    store, _ = _dict_backed_store()
    kept = _record(s3_uri=uri, bucket="test-bucket", key="test.txt",
                   human_decision="KEEP")
    store.save_record(kept)
    pipeline.memory_store = store

    out = pipeline.process_object("test.txt", skip_inference=False,
                                  execute_action=False)

    assert out["status"] == "ANALYZED"
    reloaded = store.get_record(uri)
    assert reloaded is not None
    assert reloaded.human_decision == "KEEP"
    # Fresh analysis values must still be refreshed, not frozen.
    assert reloaded.recommended_action == "archive"


def test_pipeline_cache_hit_tolerates_legacy_uppercase_action():
    """A stale uppercase recommended_action must not crash the scan.

    Observed live: demo/project-plan.txt carries recommended_action="REVIEW"
    (written before the lowercase fix) and every scan crashed rebuilding the
    cached SemanticAnalysisResult.
    """
    import hashlib

    with patch("boto3.client"):
        pipeline = SMSPipeline("test-bucket")
    content = "Hello World"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    uri = "s3://test-bucket/test.txt"
    meta = FileMetadata(
        bucket="test-bucket",
        key="test.txt",
        size_bytes=11,
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        etag="same-etag",
    )
    pipeline.inventory.collect = MagicMock(return_value=[meta])
    pipeline.reader.get_text = MagicMock(
        return_value=RetrievedContent(
            bucket="test-bucket", key="test.txt", content=content,
            content_type="text/plain", size_bytes=11,
        )
    )
    pipeline.comprehend.analyze_text = MagicMock(
        return_value={"pii_entities": [], "entities": []}
    )
    store, _ = _dict_backed_store()
    legacy = _record(s3_uri=uri, bucket="test-bucket", key="test.txt",
                     etag="same-etag", size_bytes=11,
                     content_hash=content_hash, recommended_action="review")
    # Simulate the legacy row written before the lowercase fix.
    legacy.__dict__["recommended_action"] = "REVIEW"
    store.save_record(legacy)
    pipeline.memory_store = store
    pipeline.embedding_provider = None

    out = pipeline.process_object("test.txt", skip_inference=False,
                                  execute_action=False)

    assert out["status"] == "ANALYZED"
    assert out["decision"]["action"] in ("KEEP", "ARCHIVE", "REVIEW", "TRASH")
