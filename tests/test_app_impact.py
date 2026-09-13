"""Storage & Cost Impact AppTests: hero section shows real computed values.

Fakes carry real byte sizes; the section must reflect the impact model
exactly — no invented savings, REVIEW never counted, projection labeled.
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
TIMEOUT = 90
MB = 1024 * 1024


def _record(key, size, policy, decision=None):
    return SemanticMemoryRecord(
        s3_uri=f"s3://bucket/{key}", bucket="bucket", key=key,
        etag="e", size_bytes=size, category="Reports",
        sensitivity="internal", importance_score=0.2,
        analysis_timestamp=datetime(2026, 9, 10, 10, 0, 0,
                                    tzinfo=timezone.utc),
        embedding_model="m", vector_id="v",
        recommended_action=policy, human_decision=decision)


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
        self.records[uri] = record.model_copy(
            update={"human_decision": decision})


def _fake_with(records, sizes, execute=None):
    class _Fake:
        bucket_name = "bucket"
        memory_store = _Store(records)
        action_engine = type(
            "E", (), {"execute": staticmethod(execute or _noop)})()
        inventory = type("I", (), {"collect": staticmethod(
            lambda: [FileMetadata(key=k, size_bytes=sizes[k],
                                  created_at=datetime.now(timezone.utc))
                     for k in sizes])})()
        reader = type("R", (), {"get_text": staticmethod(
            lambda meta: RetrievedContent(
                bucket="bucket", key=meta.key, content="content",
                content_type="text/plain", size_bytes=10))})()
        comprehend = type("C", (), {"analyze_text": staticmethod(
            lambda text: {"pii_entities": [], "entities": []})})()
        relationship_analyzer = type("RA", (), {"analyze": staticmethod(
            lambda *a, **k: [])})()

    return _Fake()


def _noop(request):
    return ActionResult(action=request.requested_action, key=request.key,
                        status="FAILED", message="noop")


def _records():
    return [_record("demo/keep.pdf", MB, "retain"),
            _record("demo/review.pdf", MB, "review"),
            _record("demo/old.pdf", MB, "archive")]


def _sizes():
    return {"demo/keep.pdf": MB, "demo/review.pdf": MB,
            "demo/old.pdf": MB}


def _start():
    cache_data.clear()
    at = AppTest.from_file(APP)
    at.session_state["pipeline"] = _fake_with(_records(), _sizes())
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"initial run crashed: {at.exception}"
    return at


def _blob(at):
    parts = []
    for coll in (at.markdown, at.info, at.success, at.caption):
        parts.extend(e.value for e in coll)
    parts.extend(str(m.value) for m in at.metric)
    parts.extend(m.label for m in at.metric)
    return "\n".join(parts)


def test_impact_section_shows_real_computed_values():
    at = _start()
    blob = _blob(at)
    assert "STORAGE" in blob and "COST IMPACT" in blob
    assert "3.0 MB" in blob  # 3 x 1 MB active workspace
    assert "1.0 MB" in blob  # 1 MB archive-policy potential
    assert "PROJECTED" in blob
    assert "not your AWS bill" in blob or "not your bill" in blob


def test_review_docs_not_counted_as_savings():
    from sms_agent.impact import compute_impact
    at = _start()
    blob = _blob(at)
    assert "33%" in blob  # 1 of 3 MB is archive-policy
    model = compute_impact(
        [{"Filename": "demo/old.pdf", "SizeBytes": MB, "Policy": "archive"},
         {"Filename": "demo/review.pdf", "SizeBytes": MB,
          "Policy": "review"}],
        managed_sizes=[])
    assert model["potential_bytes"] == MB


def test_chart_helper_builds_labeled_series():
    from unittest.mock import MagicMock
    from sms_agent.ui_state import render_impact_chart
    box = MagicMock()
    assert render_impact_chart(
        box, [{"month": 0, "baseline_bytes": 3 * MB,
               "managed_bytes": 2 * MB, "kind": "projected"},
              {"month": 12, "baseline_bytes": 3 * MB,
               "managed_bytes": 2 * MB, "kind": "projected"}]) is True
    spec = str(box.altair_chart.call_args[0][0].to_dict())
    assert "Without SMS" in spec and "With SMS" in spec
    assert "Projected" in spec


def test_format_mix_note_lists_real_formats():
    at = _start()
    blob = _blob(at)
    assert "3 documents" in blob
    assert "PDF" in blob
