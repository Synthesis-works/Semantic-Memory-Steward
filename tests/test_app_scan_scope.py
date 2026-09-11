"""Scan-scope AppTests: discovery follows current S3 reality.

Fake S3 inventory is fully controllable; the memory store is a real
DynamoDBMemoryStore over a dict-backed table; LLM/Comprehend are mocked
at the edges. The pipeline, policy, and app scan flow are all real.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from streamlit import cache_data
from streamlit.testing.v1 import AppTest

from sms_agent.memory import DynamoDBMemoryStore
from sms_agent.models import (
    FileMetadata,
    RetrievedContent,
    SemanticAnalysisResult,
    SemanticMemoryRecord,
)
from sms_agent.pipeline import SMSPipeline

APP = r"D:\SMS\app.py"
TIMEOUT = 120


def _analysis_for(file_key):
    return SemanticAnalysisResult(
        key=file_key, category=f"cat-of-{file_key}",
        sensitivity="internal", importance_score=0.5, confidence=0.9,
        reasoning="mocked", recommended_action="review")


def _item(key):
    return FileMetadata(key=key, size_bytes=20,
                        created_at=datetime.now(timezone.utc), etag="e1")


def _dict_store(records=()):
    box = {}
    for record in records:
        saved = record.model_dump(mode="json")
        saved["importance_score"] = record.importance_score
        saved["analysis_timestamp"] = record.analysis_timestamp.isoformat()
        box[record.s3_uri] = saved
    table = MagicMock()
    table.get_item.side_effect = lambda Key: (
        {"Item": box[Key["s3_uri"]]} if Key["s3_uri"] in box else {})
    table.put_item.side_effect = lambda Item: box.__setitem__(
        Item["s3_uri"], dict(Item))
    resource = MagicMock()
    resource.Table.return_value = table
    return DynamoDBMemoryStore(table_name="t", dynamodb_resource=resource), box


def _record(key):
    return SemanticMemoryRecord(
        s3_uri=f"s3://bucket/{key}", bucket="bucket", key=key, etag="e",
        size_bytes=20, category="old", sensitivity="internal",
        importance_score=0.5,
        analysis_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        embedding_model="none", vector_id="v", recommended_action="review")


def _pipeline(keys, store):
    with patch("boto3.client"):
        pipeline = SMSPipeline("bucket", memory_store=store)
    pipeline.inventory.collect = MagicMock(
        return_value=[_item(k) for k in keys])
    pipeline.reader.get_text = MagicMock(
        side_effect=lambda meta: RetrievedContent(
            bucket="bucket", key=meta.key, content=f"content of {meta.key}",
            content_type="text/plain", size_bytes=20))
    pipeline.comprehend.analyze_text = MagicMock(
        return_value={"pii_entities": [], "entities": []})
    return pipeline


def _run_scan(keys, records=()):
    store, box = _dict_store(records)
    pipeline = _pipeline(keys, store)
    analyze = patch("sms_agent.agent.SMSAgent.analyze_file",
                    side_effect=lambda file_key, content, metadata=None, trace=None: _analysis_for(file_key))
    analyze.start()
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = pipeline
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"initial run crashed: {at.exception}"
    for button in at.button:
        if "Run Full SMS Scan" in button.label:
            button.click().run(timeout=TIMEOUT)
            break
    else:
        analyze.stop()
        raise AssertionError("scan button not found")
    assert not at.exception, f"scan crashed: {at.exception}"
    analyze.stop()
    return at, box


def _table_rows(at):
    assert len(at.dataframe) == 1
    return at.dataframe[0].value


def _trace_events(at):
    return at.session_state["sms_trace_events"]


def test_new_doc_without_memory_is_discovered_and_persisted():
    at, box = _run_scan(["demo/new.txt"])
    frame = _table_rows(at)
    assert frame["Filename"].tolist() == ["demo/new.txt"]
    assert "cat-of-demo/new.txt" in frame["Category"].tolist()
    assert "s3://bucket/demo/new.txt" in box  # persisted to memory


def test_trash_archive_and_stale_records_excluded():
    stale = _record("trash/demo/b.txt")
    at, _ = _run_scan(
        ["demo/a.txt", "trash/demo/b.txt", "archive/demo/c.txt"],
        records=[stale])
    frame = _table_rows(at)
    assert frame["Filename"].tolist() == ["demo/a.txt"]
    assert not any("PENDING_SCAN" in str(v)
                   for v in frame["Policy"].tolist())
    discovery = next(e for e in _trace_events(at)
                     if e.get("stage") == "DISCOVERY")
    assert "1 active workspace" in discovery.get("detail", "")


def test_moved_doc_and_missing_object_stay_out():
    stale_moved = _record("demo/gone.txt")
    at, _ = _run_scan(["demo/kept.txt"], records=[stale_moved])
    frame = _table_rows(at)
    assert frame["Filename"].tolist() == ["demo/kept.txt"]


def test_counts_agree_across_trace_table_and_summary():
    at, _ = _run_scan(["demo/one.txt", "demo/two.txt"])
    frame = _table_rows(at)
    assert len(frame) == 2
    discovery = next(e for e in _trace_events(at)
                     if e.get("stage") == "DISCOVERY")
    assert "2 active workspace" in discovery.get("detail", "")
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics.get("Documents analyzed") == "2"


def test_unscanned_workspace_doc_honestly_pending():
    store, _ = _dict_store()
    pipeline = _pipeline(["demo/fresh.txt"], store)
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = pipeline
    at.run(timeout=TIMEOUT)
    frame = _table_rows(at)
    assert frame["Filename"].tolist() == ["demo/fresh.txt"]
    assert frame["Status"].tolist() == ["-"]
