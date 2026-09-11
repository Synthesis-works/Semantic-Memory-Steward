"""Document-context tests: one canonical active doc, per-doc selection,
doc-scoped results. Every test pins that KEEP and QUARANTINE can never
show contradictory stale text for the active document.
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

APP = r"D:\SMS\app.py"
KEY_A = "demo/alpha.txt"
KEY_B = "demo/beta.txt"
TIMEOUT = 90


def _record(key, category="Financial Reporting", sensitivity="confidential",
            importance=0.37, policy="review", decision=None):
    return SemanticMemoryRecord(
        s3_uri=f"s3://bucket/{key}", bucket="bucket", key=key,
        etag="e", size_bytes=52, category=category,
        sensitivity=sensitivity, importance_score=importance,
        analysis_timestamp=datetime(2026, 9, 10, 10, 0, 0,
                                    tzinfo=timezone.utc),
        embedding_model="m", vector_id="v",
        recommended_action=policy, human_decision=decision)


def _contents():
    return {KEY_A: "Alpha quarterly figures.",
            KEY_B: "Beta technical notes."}


class _Store:
    def __init__(self, records):
        self.records = {r.s3_uri: r for r in records}

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


def _fake_with(execute, keys):
    store = _Store([_record(
        k,
        category=("Technical" if k == KEY_B else "Financial Reporting"),
        sensitivity=("internal" if k == KEY_B else "confidential"),
        importance=(0.66 if k == KEY_B else 0.37)) for k in keys])

    class _Fake:
        bucket_name = "bucket"
        memory_store = store
        action_engine = type(
            "E", (), {"execute": staticmethod(execute)})()
        inventory = type("I", (), {"collect": staticmethod(
            lambda: [FileMetadata(key=k, size_bytes=52, created_at=datetime.now(
                timezone.utc)) for k in keys])})()
        reader = type("R", (), {"get_text": staticmethod(
            lambda meta: RetrievedContent(
                bucket="bucket", key=meta.key,
                content=_contents()[meta.key], content_type="text/plain",
                size_bytes=10))})()
        comprehend = type("C", (), {"analyze_text": staticmethod(
            lambda text: {"pii_entities": [], "entities": []})})()
        relationship_analyzer = type("RA", (), {"analyze": staticmethod(
            lambda *a, **k: [])})()

    return _Fake()


def _verified_keep(request):
    assert request.human_approved is True
    return ActionResult(action="KEEP", key=request.key,
                        status="VERIFIED_NO_ACTION",
                        message="No mutation required.")


def _verified_quarantine(request):
    assert request.requested_action == "QUARANTINE"
    return ActionResult(action="QUARANTINE", key=request.key,
                        status="VERIFIED",
                        message=f"Successfully quarantined to trash/{request.key}.")


def _failed_quarantine(request):
    return ActionResult(action="QUARANTINE", key=request.key,
                        status="FAILED",
                        message="Source object does not exist.")


def _start(execute, keys=(KEY_A, KEY_B)):
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = _fake_with(execute, keys)
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"initial run crashed: {at.exception}"
    return at


def _texts(elements):
    return [e.value for e in elements]


def _blob(at):
    return ("\n".join(_texts(at.markdown)) + "\n"
            + "\n".join(_texts(at.info)) + "\n"
            + "\n".join(_texts(at.success)) + "\n"
            + "\n".join(_texts(at.error)) + "\n"
            + "\n".join(_texts(at.subheader)) + "\n"
            + "\n".join(_texts(at.header)) + "\n"
            + "\n".join(_texts(at.code)))


def _click_button(at, match):
    for button in at.button:
        if match(button):
            button.click().run(timeout=TIMEOUT)
            assert not at.exception, f"crashed: {at.exception}"
            return
    raise AssertionError(
        f"no matching button; saw {[b.label for b in at.button]}")


def _open_review(at, key):
    _click_button(
        at, lambda b: b.label == "Review" and (b.key or "").endswith(key))


def _open_other_doc(at, key):
    _click_button(
        at, lambda b: (b.key or "").startswith("sms_open_")
        and (b.key or "").endswith(key))


def test_review_header_names_active_document():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    blob = _blob(at)
    assert "REVIEWING" in blob
    assert KEY_A in blob
    assert "PENDING REVIEW" in blob
    assert "CONFIDENTIAL" in blob


def test_other_docs_navigation_switches_context():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    _open_other_doc(at, KEY_B)
    blob = _blob(at)
    assert KEY_B in blob
    assert "Technical" in blob  # B's real category, not A's
    assert "PII: not detected" in blob


def test_selection_does_not_leak_across_documents():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    assert f"trash/{KEY_A}" in _blob(at)
    _open_other_doc(at, KEY_B)
    assert at.radio[0].value == "KEEP"
    blob = _blob(at)
    assert "No S3 file move" in blob
    assert "YOU ARE ABOUT TO KEEP" in blob
    assert "YOU ARE ABOUT TO QUARANTINE" not in blob
    assert not any("Confirm Quarantine" in b.label for b in at.button)


def test_switching_back_restores_per_doc_selection():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    _open_other_doc(at, KEY_B)
    _open_other_doc(at, KEY_A)
    assert at.radio[0].value == "QUARANTINE"
    assert f"trash/{KEY_A}" in _blob(at)


def test_recommendation_stays_separate_from_selection():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    before = [line for line in _blob(at).split("\n")
              if "SMS RECOMMENDATION" in line]
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    blob = _blob(at)
    assert "SMS RECOMMENDATION" in blob
    assert "YOUR DECISION" in blob
    after = [line for line in blob.split("\n")
             if "SMS RECOMMENDATION" in line]
    assert before == after


def test_button_label_follows_selection():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    assert any("Confirm Keep" in b.label for b in at.button)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    assert any("Quarantine" in b.label for b in at.button)


def test_result_scoped_to_acted_document():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    _click_button(at, lambda b: "Confirm Keep" in b.label)
    assert KEY_A in _blob(at)
    _open_other_doc(at, KEY_B)
    blob = _blob(at)
    assert not any(KEY_A in s for s in _texts(at.success)), (
        "another document's success banner must not show here")
    assert "What happened" not in blob
    assert "KEEP CONFIRMED" not in blob
    assert "No S3 file move" in blob  # B's own KEEP consequences


def test_prev_next_loads_each_documents_own_state():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    _click_button(at, lambda b: b.label == "Next →")
    assert KEY_B in _blob(at)
    assert at.radio[0].value == "KEEP"
    _click_button(at, lambda b: b.label == "← Previous")
    assert KEY_A in _blob(at)
    assert at.radio[0].value == "QUARANTINE"


def test_queue_review_opens_correct_document():
    at = _start(_verified_keep)
    _open_review(at, KEY_B)
    blob = _blob(at)
    assert KEY_B in blob
    assert "Technical" in blob


def test_resolved_review_leaves_queue_and_empty_message():
    at = _start(_verified_keep, keys=(KEY_A,))
    _open_review(at, KEY_A)
    _click_button(at, lambda b: "Confirm Keep" in b.label)
    _click_button(at, lambda b: "Back to workspace" in b.label)
    blob = _blob(at)
    assert "Needs your attention — SMS paused" not in blob
    assert "Nothing needs your attention" in blob


def test_doc_trace_excerpt_contains_only_own_events():
    at = _start(_verified_keep)
    _open_review(at, KEY_A)
    _click_button(at, lambda b: "Confirm Keep" in b.label)
    at.run(timeout=TIMEOUT)
    blob_a = _blob(at)
    assert KEY_A in blob_a
    assert "What happened" in blob_a
    _open_other_doc(at, KEY_B)
    blob_b = _blob(at)
    assert "What happened" not in blob_b
    assert "KEEP CONFIRMED" not in blob_b


def test_running_event_names_action_and_file():
    at = _start(_failed_quarantine)
    _open_review(at, KEY_A)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    _click_button(at, lambda b: "Quarantine" in b.label)
    trace = at.session_state["sms_trace_events"]
    running = [e for e in trace
               if e.get("stage") == "ACTION" and e.get("status") == "RUNNING"]
    assert len(running) == 1
    assert "QUARANTINE" in running[0].get("title", "")
    assert KEY_A in running[0].get("title", "")
    assert "Source object does not exist." in _blob(at)


def test_verified_quarantine_shows_destination_block():
    at = _start(_verified_quarantine)
    _open_review(at, KEY_A)
    at.radio[0].set_value("QUARANTINE").run(timeout=TIMEOUT)
    _click_button(at, lambda b: "Quarantine" in b.label)
    at.run(timeout=TIMEOUT)
    blob = _blob(at)
    assert "QUARANTINE VERIFIED" in blob
    assert f"trash/{KEY_A}" in blob
    assert "Destination verified" in blob
