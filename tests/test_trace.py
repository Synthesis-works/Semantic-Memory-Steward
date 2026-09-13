"""Tests for the execution-event trace (sms_agent.trace).

The trace is the testable seam for "real execution -> real events ->
human-readable activity". Every test below pins honesty properties:
events exist only because code recorded them, failures stay failures,
waiting stays waiting, and nothing document-specific is hardcoded.
"""
from datetime import datetime

from sms_agent.trace import (
    ExecutionEvent,
    TraceRecorder,
    render_trace_lines,
)


def test_recorded_event_carries_full_structure():
    rec = TraceRecorder()
    event = rec.record(
        stage="CONTENT",
        title="Read demo/a.txt",
        detail="142 chars",
        status="SUCCESS",
        document="demo/a.txt",
    )
    assert isinstance(event, ExecutionEvent)
    assert event.stage == "CONTENT"
    assert event.title == "Read demo/a.txt"
    assert event.detail == "142 chars"
    assert event.status == "SUCCESS"
    assert event.document == "demo/a.txt"
    assert isinstance(event.timestamp, datetime)
    assert rec.events == [event]


def test_succeed_helper_marks_success():
    rec = TraceRecorder()
    event = rec.succeed(stage="POLICY", title="Decision: REVIEW",
                        detail="human judgment required")
    assert event.status == "SUCCESS"


def test_failed_operations_produce_failed_events_with_reason():
    rec = TraceRecorder()
    event = rec.fail(stage="ENRICHMENT", title="AWS Comprehend",
                     detail="Enrichment failed: quota exceeded")
    assert event.status == "FAILED"
    assert "quota exceeded" in event.detail
    lines = render_trace_lines(rec.events)
    assert any("Enrichment failed" in line for line in lines)
    assert not any(line.startswith("SUCCESS") for line in lines)


def test_human_approval_produces_waiting_state():
    rec = TraceRecorder()
    event = rec.wait(stage="HUMAN_APPROVAL", title="Human approval required",
                     detail="SMS paused before taking action.",
                     document="demo/a.txt")
    assert event.status == "WAITING"
    lines = render_trace_lines(rec.events)
    assert any("paused" in line.lower() or "waiting" in line.lower()
               for line in lines)


def test_render_marks_statuses_honestly():
    rec = TraceRecorder()
    rec.succeed(stage="DISCOVERY", title="S3 Scanner", detail="Found 4 documents")
    rec.record(stage="AI_ANALYSIS", title="Semantic analysis",
               detail="Analyzing document...", status="RUNNING")
    rec.fail(stage="MEMORY", title="Semantic Memory", detail="put_item failed")
    lines = render_trace_lines(rec.events)
    assert len(lines) == 3
    assert lines[0].startswith("✓")
    assert "RUNNING" in lines[1] or "◉" in lines[1]
    assert lines[2].startswith("✗")


def test_empty_trace_renders_nothing():
    assert render_trace_lines([]) == []
    assert TraceRecorder().events == []


def test_trace_serializes_for_session_state():
    rec = TraceRecorder()
    rec.succeed(stage="CONTENT", title="Read demo/a.txt", document="demo/a.txt")
    rec.wait(stage="HUMAN_APPROVAL", title="Human approval required")
    data = rec.to_dicts()
    assert isinstance(data, list) and len(data) == 2
    restored = TraceRecorder.from_dicts(data)
    assert [e.status for e in restored.events] == ["SUCCESS", "WAITING"]
    assert [e.title for e in restored.events] == [
        "Read demo/a.txt", "Human approval required"]


def test_trace_module_has_no_hardcoded_demo_filenames():
    import pathlib
    text = pathlib.Path("src/sms_agent/trace.py").read_text(encoding="utf-8")
    for name in ("employee-contacts.txt", "financial-report.txt",
                 "old-project-log.txt", "project-plan.txt"):
        assert name not in text


def test_no_chain_of_thought_fields_exist():
    fields = set(ExecutionEvent.model_fields)
    assert "reasoning" not in fields
    assert "thought" not in fields
    assert "chain_of_thought" not in fields
