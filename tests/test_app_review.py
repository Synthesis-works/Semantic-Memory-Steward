"""Review-view interaction tests (real app.py, fakes only at AWS edges).

Covers: opening the deep review, real values shown, descriptive
action buttons with consequences, verified KEEP, visible FAILED,
rerun persistence, and trace approval/verified events.
"""
from datetime import datetime, timezone

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
KEY = "demo/custom-review.txt"
URI = f"s3://bucket/{KEY}"
TIMEOUT = 90


def _record(**kwargs):
    base = dict(
        s3_uri=URI,
        bucket="bucket",
        key=KEY,
        etag="etag1",
        size_bytes=52,
        category="Financial Reporting",
        sensitivity="confidential",
        importance_score=0.37,
        analysis_timestamp=datetime(2026, 9, 10, 10, 0, 0,
                                    tzinfo=timezone.utc),
        embedding_model="m",
        vector_id="v",
        recommended_action="review",
    )
    base.update(kwargs)
    return SemanticMemoryRecord(**base)


class _Store:
    def __init__(self):
        self.records = {URI: _record()}

    def get_record(self, uri):
        return self.records.get(uri)

    def save_record(self, record):
        self.records[record.s3_uri] = record

    def record_human_decision(self, uri, decision):
        record = self.records.get(uri)
        if record is None:
            raise ValueError("no record")
        if decision != "KEEP":
            raise ValueError("unsupported decision")
        self.records[uri] = record.model_copy(
            update={"human_decision": decision})


class _Fake:
    bucket_name = "bucket"
    memory_store = None
    action_engine = None

    def __init__(self, execute):
        self.memory_store = _Store()
        self.action_engine = type(
            "E", (), {"execute": staticmethod(execute)})()


def _fake_with(execute):
    fake = _Fake(execute)
    item = FileMetadata(key=KEY, size_bytes=52,
                        created_at=datetime.now(timezone.utc))
    fake.inventory = type("I", (), {"collect": staticmethod(
        lambda: [item])})()
    fake.reader = type("R", (), {"get_text": staticmethod(
        lambda meta: RetrievedContent(
            bucket="bucket", key=KEY, content="Q3 Revenue up",
            content_type="text/plain", size_bytes=13))})()
    fake.comprehend = type("C", (), {"analyze_text": staticmethod(
        lambda text: {"pii_entities": [], "entities": []})})()
    fake.relationship_analyzer = type("RA", (), {"analyze": staticmethod(
        lambda *a, **k: [])})()
    return fake


def _verified_keep(request):
    return ActionResult(action="KEEP", key=request.key,
                        status="VERIFIED_NO_ACTION",
                        message="No mutation required.")


def _failed_quarantine(request):
    return ActionResult(action="QUARANTINE", key=request.key,
                        status="FAILED",
                        message="Source object does not exist.")


def _open_review(execute):
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = _fake_with(execute)
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"initial run crashed: {at.exception}"
    for button in at.button:
        if button.label == "Review":
            button.click().run(timeout=TIMEOUT)
            break
    else:
        raise AssertionError("no Review button for the REVIEW document")
    assert not at.exception
    return at


def _texts(elements):
    return [e.value for e in elements]


def _blob(at):
    return ("\n".join(_texts(at.markdown)) + "\n"
            + "\n".join(_texts(at.info)) + "\n"
            + "\n".join(_texts(at.success)) + "\n"
            + "\n".join(_texts(at.error)) + "\n"
            + "\n".join(_texts(at.subheader)) + "\n"
            + "\n".join(_texts(at.code)))


def test_review_view_shows_real_document_values():
    at = _open_review(_verified_keep)
    blob = _blob(at)
    assert KEY in blob
    assert "Financial Reporting" in blob
    assert "confidential" in blob
    assert "0.37" in blob
    assert "REVIEW REQUIRED" in blob
    assert "Q3 Revenue up" in blob  # bounded content preview
    assert "PII: not detected" in blob


def test_review_action_buttons_are_descriptive():
    at = _open_review(_verified_keep)
    labels = [b.label for b in at.button]
    assert any("Confirm Keep" in label for label in labels), labels
    assert not any(label == "Confirm & Execute" for label in labels), labels
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    labels = [b.label for b in at.button]
    assert any("Quarantine" in label for label in labels), labels
    blob = _blob(at)
    assert "YOU ARE ABOUT TO" in blob
    assert f"trash/{KEY}" in blob


def test_review_keep_confirms_verified_result_and_persists():
    at = _open_review(_verified_keep)
    for button in at.button:
        if "Confirm Keep" in button.label:
            button.click().run(timeout=TIMEOUT)
    assert not at.exception
    blob = _blob(at)
    assert "KEEP" in blob and KEY in blob
    assert "verif" in blob.lower()
    assert "No mutation required." in blob
    at.run(timeout=TIMEOUT)
    blob = _blob(at)
    assert "KEEP" in blob and KEY in blob  # survives rerun


def test_review_failed_action_stays_failed():
    at = _open_review(_failed_quarantine)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    for button in at.button:
        if "Quarantine" in button.label:
            button.click().run(timeout=TIMEOUT)
    assert not at.exception
    at.run(timeout=TIMEOUT)
    blob = _blob(at)
    assert "Source object does not exist." in blob
    assert "FAILED" in blob


def test_review_trace_records_approval_and_verified_events():
    at = _open_review(_verified_keep)
    for button in at.button:
        if "Confirm Keep" in button.label:
            button.click().run(timeout=TIMEOUT)
    try:
        trace = at.session_state["sms_trace_events"]
    except KeyError:
        trace = []
    actions = [e for e in trace if e.get("stage") == "ACTION"]
    assert [e.get("status") for e in actions] == ["RUNNING", "SUCCESS"]
    assert "Approval received" in actions[0].get("title", "")


def test_charts_render_from_real_rows():
    from unittest.mock import MagicMock
    from sms_agent.ui_state import (
        render_category_bars,
        render_donut,
    )
    box = MagicMock()
    assert render_donut(box, {"REVIEW": 1, "ARCHIVE": 1},
                        "Policy decisions") is True
    assert render_category_bars(box, {"Financial Reporting": 1},
                                "Categories") is True
    assert box.altair_chart.call_count == 2
    specs = " ".join(
        str(call.args[0].to_dict())
        for call in box.altair_chart.call_args_list)
    assert "Financial Reporting" in specs
    assert "REVIEW" in specs
    assert "ARCHIVE" in specs
    empty_box = MagicMock()
    assert render_donut(empty_box, {}, "Empty") is False
    empty_box.altair_chart.assert_not_called()


def test_no_fake_trend_captions_in_app():
    import pathlib
    text = pathlib.Path(r"D:\SMS\app.py").read_text(encoding="utf-8")
    assert "↑" not in text
    assert "↓" not in text
    assert "last week" not in text.lower()
    assert "last month" not in text.lower()
