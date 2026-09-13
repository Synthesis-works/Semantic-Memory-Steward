"""End-to-end regression tests for the dashboard approval flow (real app.py).

Observed failure: REVIEW -> KEEP -> Confirm & Execute appeared to do nothing.
These tests drive the actual Streamlit app and assert the truthful,
persistent state transitions the fix must produce.

The fake pipeline exercises the same seams as production (inventory,
memory store with get/save/record_human_decision, action engine, reader,
comprehend). No network is touched.
"""
from datetime import datetime, timezone

import pytest
from streamlit import cache_data
from streamlit.testing.v1 import AppTest

from sms_agent.models import (
    ActionResult,
    FileMetadata,
    RetrievedContent,
    SemanticMemoryRecord,
)

import os

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
KEY = "demo/review-doc.txt"
URI = f"s3://bucket/{KEY}"
TIMEOUT = 60


def _record(**kwargs):
    base = dict(
        s3_uri=URI,
        bucket="bucket",
        key=KEY,
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


class FakeMemoryStore:
    """Dict-backed stand-in for DynamoDBMemoryStore (same method surface)."""

    def __init__(self, records=None):
        records = records if records is not None else [_record()]
        self.records = {r.s3_uri: r for r in records}

    def get_record(self, s3_uri):
        return self.records.get(s3_uri)

    def save_record(self, record):
        self.records[record.s3_uri] = record

    def record_human_decision(self, s3_uri, decision):
        record = self.records.get(s3_uri)
        if record is None:
            raise ValueError(f"record_human_decision: no record for {s3_uri!r}")
        if decision != "KEEP":
            raise ValueError(f"record_human_decision: unsupported decision {decision!r}")
        self.records[s3_uri] = record.model_copy(update={"human_decision": decision})


class FakePipeline:
    """Mirrors the SMSPipeline attributes app.py touches."""

    def __init__(self, execute, keys=None, records=None):
        self.bucket_name = "bucket"
        self.memory_store = FakeMemoryStore(records)
        self.action_engine = _FakeEngine(execute)
        self.inventory = _FakeInventory(keys or [KEY])
        self.reader = _FakeReader()
        self.comprehend = _FakeComprehend()


class _FakeEngine:
    def __init__(self, execute):
        self._execute = execute

    def execute(self, request):
        return self._execute(request)


class _FakeInventory:
    def __init__(self, keys):
        self.keys = list(keys)

    def collect(self):
        return [FileMetadata(key=k, size_bytes=100,
                             created_at=datetime.now(timezone.utc))
                for k in self.keys]


class _FakeReader:
    def get_text(self, meta):
        return RetrievedContent(bucket="bucket", key=KEY, content="hello",
                                content_type="text/plain", size_bytes=5)


class _FakeComprehend:
    def analyze_text(self, text):
        return {"pii_entities": [], "entities": []}


def _verified_keep(request):
    assert request.requested_action == "KEEP"
    assert request.human_approved is True
    return ActionResult(action="KEEP", key=request.key,
                        status="VERIFIED_NO_ACTION",
                        message="No mutation required.")


def _failed_quarantine(request):
    return ActionResult(action="QUARANTINE", key=request.key, status="FAILED",
                        message="Source object does not exist.")


@pytest.fixture
def keep_app():
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = FakePipeline(_verified_keep)
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"initial run crashed: {at.exception}"
    at.selectbox[0].set_value(KEY).run(timeout=TIMEOUT)
    at.radio[0].set_value("KEEP").run(timeout=TIMEOUT)
    for button in at.button:
        if "Confirm" in button.label:
            button.click().run(timeout=TIMEOUT)
    assert not at.exception, f"confirm run crashed: {at.exception}"
    return at


def _texts(elements):
    return [e.value for e in elements]


def test_keep_confirm_execute_shows_verified_result(keep_app):
    at = keep_app
    shown = _texts(at.success) + _texts(at.error) + _texts(at.info)
    assert any("KEEP" in t and KEY in t for t in shown), (
        f"no visible KEEP result for {KEY}; success={_texts(at.success)} "
        f"error={_texts(at.error)}"
    )
    assert any("verif" in t.lower() for t in shown), (
        "KEEP result must state the outcome was verified"
    )


def test_keep_result_survives_rerun(keep_app):
    keep_app.run(timeout=TIMEOUT)
    assert not keep_app.exception
    shown = (_texts(keep_app.success) + _texts(keep_app.error)
             + _texts(keep_app.info))
    assert any("KEEP" in t and KEY in t for t in shown), (
        "KEEP result must survive a rerun"
    )


def test_keep_marks_document_kept_in_table(keep_app):
    at = keep_app
    frames = [df.value for df in at.dataframe]
    assert frames, "expected the document table to render"
    statuses = set()
    for frame in frames:
        if "Status" in frame.columns:
            statuses.update(str(v) for v in frame["Status"].tolist())
    assert "KEPT" in statuses, f"expected a KEPT row, saw statuses={statuses}"


def test_failed_action_stays_visible_with_reason():
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = FakePipeline(_failed_quarantine)
    at.run(timeout=TIMEOUT)
    at.selectbox[0].set_value(KEY).run(timeout=TIMEOUT)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    for button in at.button:
        if "Confirm" in button.label:
            button.click().run(timeout=TIMEOUT)
    assert not at.exception
    at.run(timeout=TIMEOUT)
    shown = _texts(at.error) + _texts(at.success)
    assert any("Source object does not exist." in t for t in shown), (
        "failed action must keep showing its reason after rerun"
    )
    assert not any("verif" in t.lower() and "success" in t.lower() for t in shown), (
        "a failed action must never be presented as verified success"
    )


ARCHIVE_KEY = "demo/old-log.txt"
ARCHIVE_URI = f"s3://bucket/{ARCHIVE_KEY}"


def _archive_record():
    return _record(s3_uri=ARCHIVE_URI, bucket="bucket", key=ARCHIVE_KEY,
                   category="technical", sensitivity="public",
                   importance_score=0.2, recommended_action="archive")


def _run_to_inspector(execute, keys, records):
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = FakePipeline(execute, keys=keys,
                                                records=records)
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"initial run crashed: {at.exception}"
    return at


def test_quarantine_success_names_verified_destination():
    def _exec(request):
        assert request.requested_action == "QUARANTINE"
        assert request.human_approved is True
        return ActionResult(action="QUARANTINE", key=request.key,
                            status="VERIFIED",
                            message=f"Successfully quarantined to trash/{request.key}.")

    at = _run_to_inspector(_exec, [KEY], [_record()])
    at.selectbox[0].set_value(KEY).run(timeout=TIMEOUT)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    for button in at.button:
        if "Confirm" in button.label:
            button.click().run(timeout=TIMEOUT)
    assert not at.exception
    at.run(timeout=TIMEOUT)
    shown = _texts(at.success) + _texts(at.error) + _texts(at.info)
    assert any("QUARANTINE" in t and "trash/" + KEY in t and "verif" in t.lower()
               for t in shown), (
        f"expected verified QUARANTINE outcome naming trash/{KEY}; "
        f"success={_texts(at.success)} error={_texts(at.error)}"
    )


def test_archive_recommendation_offers_verified_execute():
    seen = {}

    def _exec(request):
        seen["action"] = request.requested_action
        seen["risk"] = request.risk
        seen["approved"] = request.human_approved
        assert request.requested_action == "ARCHIVE"
        return ActionResult(action="ARCHIVE", key=request.key,
                            status="VERIFIED",
                            message=f"Successfully archived to archive/{request.key}.")

    at = _run_to_inspector(_exec, [ARCHIVE_KEY], [_archive_record()])
    at.selectbox[0].set_value(ARCHIVE_KEY).run(timeout=TIMEOUT)
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Archive" in label for label in labels), (
        "expected an ARCHIVE execute control for a policy-archive document; "
        f"buttons={labels}"
    )
    for button in at.button:
        if "Archive" in button.label:
            button.click().run(timeout=TIMEOUT)
    assert not at.exception
    assert seen.get("action") == "ARCHIVE"
    assert seen.get("risk") == "LOW", "policy ARCHIVE implies LOW risk"
    assert seen.get("approved") is True
    at.run(timeout=TIMEOUT)
    shown = _texts(at.success) + _texts(at.error) + _texts(at.info)
    assert any("ARCHIVE" in t and "archive/" + ARCHIVE_KEY in t
               and "verif" in t.lower() for t in shown), (
        f"expected verified ARCHIVE outcome naming archive/{ARCHIVE_KEY}; "
        f"success={_texts(at.success)} error={_texts(at.error)}"
    )


def test_workspace_summary_reflects_real_records():
    review = _record()
    kept = _record(s3_uri="s3://bucket/demo/kept.txt", key="demo/kept.txt",
                   human_decision="KEEP")
    old = _record(s3_uri="s3://bucket/demo/old.txt", key="demo/old.txt",
                  sensitivity="public", importance_score=0.2,
                  recommended_action="archive")
    keys = ["demo/review-doc.txt", "demo/kept.txt", "demo/old.txt"]
    at = _run_to_inspector(lambda req: _verified_keep(req), keys,
                           [review, kept, old])
    shown = (_texts(at.markdown) + _texts(at.info) + _texts(at.success)
             + _texts(at.header) + _texts(at.subheader))
    blob = "\n".join(shown)
    assert "Needs your attention" in blob
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics.get("Needs your attention") == "1", f"metrics={metrics}"
    assert metrics.get("Safe to keep") == "1", f"metrics={metrics}"
    assert metrics.get("Cleanup candidates") == "1", f"metrics={metrics}"


def test_provider_status_is_secondary_but_not_hidden():
    at = _run_to_inspector(lambda req: _verified_keep(req), [KEY], [_record()])
    top_sidebar = _texts(at.sidebar.markdown)
    assert any("fallback" in t.lower() and "bedrock" in t.lower()
               for t in top_sidebar), (
        "a one-line honest provider summary must stay visible"
    )
    expanders = {e.label: [m.value for m in e.markdown]
                 for e in at.sidebar.expander}
    tech_labels = [label for label in expanders if "Technical" in label]
    assert tech_labels, f"expected a Technical details expander; saw {list(expanders)}"
    assert any("Bedrock" in label for label in tech_labels), (
        "the Bedrock limitation must remain disclosed, not hidden"
    )
    tech_text = expanders[tech_labels[0]]
    assert any("SageMaker" in t for t in tech_text), (
        "full infrastructure diagnostics must live inside the expander"
    )
    assert any("Bedrock" in t for t in tech_text)
