"""Pipeline execution-trace tests: events must mirror real stage behavior.

Uses the real SMSPipeline with mocks only at the AWS/LLM boundary
(S3 client, Strands analyze_file, Comprehend). If a stage is skipped
(cached analysis) the trace must say so — never a fabricated "analyzing".
"""
import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from sms_agent.models import (
    FileMetadata,
    RetrievedContent,
    SemanticAnalysisResult,
    SemanticMemoryRecord,
)
from sms_agent.pipeline import SMSPipeline
from sms_agent.trace import TraceRecorder


def _meta(**kwargs):
    base = dict(
        bucket="test-bucket",
        key="test.txt",
        size_bytes=11,
        created_at="2020-01-01T00:00:00Z",
        last_accessed_at="2020-06-01T00:00:00Z",
        etag="etag1",
    )
    base.update(kwargs)
    return FileMetadata(**base)


def _content(text="Hello World"):
    return RetrievedContent(
        bucket="test-bucket", key="test.txt", content=text,
        content_type="text/plain", size_bytes=len(text),
    )


def _analysis(**kwargs):
    base = dict(
        key="test.txt", category="technical", sensitivity="public",
        importance_score=0.3, confidence=0.9, reasoning="stale log",
        recommended_action="archive",
    )
    base.update(kwargs)
    return SemanticAnalysisResult(**base)


def _pipeline(meta, content, analysis=None, memory_store=None):
    with patch("boto3.client"):
        pipeline = SMSPipeline("test-bucket")
    pipeline.inventory.collect = MagicMock(return_value=[meta])
    pipeline.reader.get_text = MagicMock(return_value=content)
    if analysis is not None:
        patcher = patch("sms_agent.agent.SMSAgent.analyze_file",
                        return_value=analysis)
        patcher.start()
        pipeline._trace_patcher = patcher
    pipeline.comprehend.analyze_text = MagicMock(
        return_value={"pii_entities": [],
                      "entities": [{"type": "PERSON", "text": "Alice"}]})
    pipeline.memory_store = memory_store
    return pipeline


def _stages(rec):
    return [e.stage for e in rec.events]


def test_trace_records_real_stage_values():
    pipeline = _pipeline(_meta(), _content(), _analysis(),
                         memory_store=MagicMock())
    try:
        rec = TraceRecorder()
        out = pipeline.process_object("test.txt", skip_inference=False,
                                      execute_action=False, trace=rec)
    finally:
        pipeline._trace_patcher.stop()
    assert out["status"] == "ANALYZED"
    # Fresh analysis emits started + completed; every other stage one event.
    assert _stages(rec) == ["CONTENT", "AI_ANALYSIS", "AI_ANALYSIS",
                            "ENRICHMENT", "IMPORTANCE", "POLICY", "MEMORY"]
    assert [e.status for e in rec.events] == ["SUCCESS", "RUNNING", "SUCCESS",
                                              "SUCCESS", "SUCCESS", "SUCCESS",
                                              "SUCCESS"]
    by_stage = {}
    for e in rec.events:
        by_stage[e.stage] = e  # completed event overwrites the started one
    assert "technical" in by_stage["AI_ANALYSIS"].detail
    assert "PERSON" in by_stage["ENRICHMENT"].detail
    assert "PII: none" in by_stage["ENRICHMENT"].detail
    assert "Score:" in by_stage["IMPORTANCE"].detail
    assert "ARCHIVE" in by_stage["POLICY"].detail
    assert "persisted" in by_stage["MEMORY"].detail.lower()
    pipeline.memory_store.save_record.assert_called_once()


def test_trace_cache_hit_reuses_honestly():
    content = _content()
    content_hash = hashlib.sha256(content.content.encode()).hexdigest()
    existing = SemanticMemoryRecord(
        s3_uri="s3://test-bucket/test.txt", bucket="test-bucket",
        key="test.txt", content_hash=content_hash, etag="etag1",
        size_bytes=11, category="technical", sensitivity="public",
        importance_score=0.2,
        analysis_timestamp=datetime.now(timezone.utc),
        embedding_model="none", vector_id="v", recommended_action="archive",
    )
    store = MagicMock()
    store.get_record.return_value = existing
    pipeline = _pipeline(_meta(), content, analysis=None, memory_store=store)
    with patch("sms_agent.agent.SMSAgent.analyze_file") as mock_analyze:
        rec = TraceRecorder()
        out = pipeline.process_object("test.txt", skip_inference=False,
                                      execute_action=False, trace=rec)
    assert out["status"] == "ANALYZED"
    mock_analyze.assert_not_called()
    store.save_record.assert_not_called()
    ai_event = next(e for e in rec.events if e.stage == "AI_ANALYSIS")
    assert "cach" in ai_event.detail.lower()  # reused, not re-analyzed
    mem_event = next(e for e in rec.events if e.stage == "MEMORY")
    assert "reus" in mem_event.detail.lower()


def test_trace_enrichment_failure_is_failed_not_faked():
    pipeline = _pipeline(_meta(), _content(), _analysis(),
                         memory_store=MagicMock())
    pipeline.comprehend.analyze_text = MagicMock(
        side_effect=RuntimeError("Comprehend quota exceeded"))
    try:
        import pytest
        rec = TraceRecorder()
        with pytest.raises(RuntimeError, match="Comprehend quota exceeded"):
            pipeline.process_object("test.txt", skip_inference=False,
                                    execute_action=False, trace=rec)
    finally:
        pipeline._trace_patcher.stop()
    failed = [e for e in rec.events if e.status == "FAILED"]
    assert len(failed) == 1
    assert failed[0].stage == "ENRICHMENT"
    assert "quota exceeded" in failed[0].detail


def test_trace_review_decision_waits_for_human():
    analysis = _analysis(sensitivity="confidential", importance_score=0.8,
                         recommended_action="review")
    pipeline = _pipeline(_meta(), _content(), analysis,
                         memory_store=MagicMock())
    try:
        rec = TraceRecorder()
        out = pipeline.process_object("test.txt", skip_inference=False,
                                      execute_action=False, trace=rec)
    finally:
        pipeline._trace_patcher.stop()
    assert out["decision"]["action"] == "REVIEW"
    waiting = [e for e in rec.events if e.status == "WAITING"]
    assert len(waiting) == 1
    assert waiting[0].stage == "HUMAN_APPROVAL"
    assert not [e for e in rec.events if e.stage == "ACTION"]


def test_trace_quarantine_verified_action_success():
    mock_s3 = MagicMock()
    state = {"test.txt": True, "trash/test.txt": False}

    def mock_head(Bucket, Key):
        from botocore.exceptions import ClientError
        if not state.get(Key):
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def mock_copy(CopySource, Bucket, Key):
        state[Key] = True

    def mock_delete(Bucket, Key):
        state[Key] = False

    mock_s3.head_object.side_effect = mock_head
    mock_s3.copy_object.side_effect = mock_copy
    mock_s3.delete_object.side_effect = mock_delete
    mock_s3.get_paginator.return_value.paginate.return_value = [{
        "Contents": [{"Key": "test.txt",
                      "LastModified": datetime(2020, 1, 1, tzinfo=timezone.utc),
                      "Size": 11, "ETag": '"etag1"'}]}]
    mock_body = MagicMock()
    mock_body.read.return_value = b"Hello World"
    mock_s3.get_object.return_value = {"Body": mock_body,
                                       "ContentType": "text/plain"}

    with patch("boto3.client"):
        pipeline = SMSPipeline("test-bucket", s3_client=mock_s3)
    pipeline.comprehend.analyze_text = MagicMock(
        return_value={"pii_entities": [], "entities": []})
    analysis = _analysis(importance_score=0.1, recommended_action="review")
    with patch("sms_agent.agent.SMSAgent.analyze_file",
               return_value=analysis):
        pipeline.memory_store = MagicMock()
        pipeline.memory_store.get_record.return_value = None
        rec = TraceRecorder()
        out = pipeline.process_object("test.txt", skip_inference=False,
                                      execute_action=True, human_approved=True,
                                      trace=rec)
    assert out["status"] == "COMPLETED"
    assert out["action_result"]["status"] == "VERIFIED"
    actions = [e for e in rec.events if e.stage == "ACTION"]
    assert [e.status for e in actions] == ["RUNNING", "SUCCESS"]
    assert "trash/test.txt" in actions[1].detail
