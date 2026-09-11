"""Dashboard LIVE ACTIVITY tests (real app.py + real pipeline, mocked edges).

The scan button drives the real SMSPipeline with mocks only at the
AWS/LLM boundary, so every asserted trace event comes from actual
instrumented execution — not from canned UI fixtures.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from streamlit import cache_data
from streamlit.testing.v1 import AppTest

from sms_agent.models import SemanticAnalysisResult
from sms_agent.pipeline import SMSPipeline

APP = r"D:\SMS\app.py"
KEY = "demo/a.txt"
TIMEOUT = 90


def _analysis(**kwargs):
    base = dict(
        key=KEY, category="Financial Reporting", sensitivity="confidential",
        importance_score=0.37, confidence=0.9, reasoning="Quarterly figures.",
        recommended_action="review",
    )
    base.update(kwargs)
    return SemanticAnalysisResult(**base)


def _real_pipeline(analysis):
    """Real pipeline; S3/memory/LLM/Comprehend stand-ins at the edges."""
    mock_s3 = MagicMock()
    dt = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
    mock_s3.get_paginator.return_value.paginate.return_value = [{
        "Contents": [{"Key": KEY, "LastModified": dt, "Size": 12,
                      "ETag": '"etag"'}]}]
    mock_body = MagicMock()
    mock_body.read.return_value = b"Q3 revenue up."
    mock_s3.get_object.return_value = {"Body": mock_body,
                                       "ContentType": "text/plain"}
    with patch("boto3.client"):
        pipeline = SMSPipeline(bucket_name="bucket", s3_client=mock_s3)
    pipeline.comprehend.analyze_text = MagicMock(
        return_value={"pii_entities": [],
                      "entities": [{"type": "PERSON", "text": "Alice"}]})
    pipeline.memory_store = MagicMock()
    pipeline.memory_store.get_record.return_value = None
    patcher = patch("sms_agent.agent.SMSAgent.analyze_file",
                    return_value=analysis)
    patcher.start()
    pipeline._trace_patcher = patcher
    return pipeline


def _run_with_pipeline(pipeline):
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = pipeline
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"initial run crashed: {at.exception}"
    return at


def _all_text(at):
    return ("\n".join(m.value for m in at.markdown)
            + "\n" + "\n".join(i.value for i in at.info)
            + "\n" + "\n".join(s.value for s in at.success)
            + "\n" + "\n".join(e.value for e in at.error))


def _click_scan(at):
    for button in at.button:
        if "Run Full SMS Scan" in button.label:
            button.click().run(timeout=TIMEOUT)
            return
    raise AssertionError("scan button not found")


def test_scan_shows_live_activity_with_real_values():
    pipeline = _real_pipeline(_analysis())
    try:
        at = _run_with_pipeline(pipeline)
        _click_scan(at)
    finally:
        pipeline._trace_patcher.stop()
    assert not at.exception, f"scan crashed: {at.exception}"
    blob = _all_text(at)
    assert "LIVE ACTIVITY" in blob
    assert f"Read {KEY}" in blob
    assert "AWS Comprehend" in blob
    assert "Financial Reporting" in blob
    assert "Policy Engine" in blob
    assert "REVIEW" in blob


def test_trace_survives_rerun_and_marks_waiting():
    pipeline = _real_pipeline(_analysis())
    try:
        at = _run_with_pipeline(pipeline)
        _click_scan(at)
        at.run(timeout=TIMEOUT)
    finally:
        pipeline._trace_patcher.stop()
    blob = _all_text(at)
    assert f"Read {KEY}" in blob
    assert "Human approval required" in blob
    assert "Scan complete in" in blob


def test_second_scan_replaces_trace():
    pipeline = _real_pipeline(_analysis())
    try:
        at = _run_with_pipeline(pipeline)
        _click_scan(at)
        first = _all_text(at).count(f"Read {KEY}")
        _click_scan(at)
        second = _all_text(at).count(f"Read {KEY}")
    finally:
        pipeline._trace_patcher.stop()
    assert first >= 1
    assert second == first, "a fresh scan must replace the old trace"


def test_action_failure_records_action_event():
    from datetime import datetime, timezone as _tz
    from sms_agent.models import (
        ActionResult as _ActionResult,
        FileMetadata as _FileMetadata,
        RetrievedContent as _RetrievedContent,
        SemanticMemoryRecord as _SemanticRecord,
    )

    record = _SemanticRecord(
        s3_uri="s3://bucket/demo/review-doc.txt", bucket="bucket",
        key="demo/review-doc.txt", etag="e", size_bytes=10,
        category="technical", sensitivity="confidential",
        importance_score=0.8,
        analysis_timestamp=datetime.now(_tz.utc), embedding_model="m",
        vector_id="v", recommended_action="review")

    class _Store:
        def get_record(self, uri):
            return record

        def save_record(self, r):
            pass

        def record_human_decision(self, uri, decision):
            pass

    class _Fake:
        bucket_name = "bucket"
        memory_store = _Store()
        inventory = None
        reader = None
        comprehend = None

        class action_engine:
            @staticmethod
            def execute(request):
                return _ActionResult(
                    action="QUARANTINE", key=request.key, status="FAILED",
                    message="Source object does not exist.")

    fake = _Fake()
    fake.inventory = type("I", (), {"collect": staticmethod(
        lambda: [_FileMetadata(key="demo/review-doc.txt", size_bytes=10,
                               created_at=datetime.now(_tz.utc))])})()
    fake.reader = type(
        "R", (), {"get_text": staticmethod(
            lambda meta: _RetrievedContent(
                bucket="bucket", key="demo/review-doc.txt", content="hi",
                content_type="text/plain", size_bytes=2))})()
    fake.comprehend = type(
        "C", (), {"analyze_text": staticmethod(
            lambda text: {"pii_entities": [], "entities": []})})()

    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = fake
    at.run(timeout=TIMEOUT)
    at.selectbox[0].set_value("demo/review-doc.txt").run(timeout=TIMEOUT)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    for button in at.button:
        if "Confirm" in button.label:
            button.click().run(timeout=TIMEOUT)
    assert not at.exception
    at.run(timeout=TIMEOUT)
    blob = _all_text(at)
    assert "Source object does not exist." in blob
    try:
        trace = at.session_state["sms_trace_events"]
    except KeyError:
        trace = []
    actions = [e for e in trace if e.get("stage") == "ACTION"]
    assert actions, "expected ACTION events in the persisted session trace"
    assert actions[-1].get("status") == "FAILED"
