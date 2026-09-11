"""Unit tests for the deep-review helpers (sms_agent.ui_state).

Every assertion pins honesty: the review model mirrors real record
values, previews are bounded, consequences name real destinations,
charts count only current rows, and nothing is hardcoded to fixtures.
"""
from sms_agent.ui_state import (
    action_consequences,
    build_review_model,
    category_counts,
    policy_counts,
    preview_content,
    sensitivity_counts,
)


def _review_kwargs(**kwargs):
    base = dict(
        key="demo/custom-doc.txt",
        category="Financial Reporting",
        sensitivity="confidential",
        importance=0.37,
        policy="review",
        human_decision=None,
        analysis_timestamp="2026-09-10T10:00:00+00:00",
        size_bytes=52,
        content_chars=52,
        preview="Q3 Revenue up",
        enrichment={"pii_entities": [], "entities": []},
        relationships=[],
    )
    base.update(kwargs)
    return base


def test_review_model_mirrors_real_values():
    model = build_review_model(**_review_kwargs())
    assert model["key"] == "demo/custom-doc.txt"
    assert model["category"] == "Financial Reporting"
    assert model["sensitivity"] == "confidential"
    assert model["importance"] == 0.37
    assert model["policy"] == "review"
    assert model["status"] == "PENDING_REVIEW"
    assert model["content_chars"] == 52


def test_review_evidence_cites_real_values():
    model = build_review_model(**_review_kwargs(
        enrichment={"pii_entities": [],
                    "entities": [{"type": "PERSON", "text": "Alice"}]}))
    blob = "\n".join(model["evidence"])
    assert "confidential" in blob
    assert "0.37" in blob
    assert "Financial Reporting" in blob
    assert "PERSON" in blob
    assert "PII: not detected" in blob


def test_review_missing_enrichment_is_graceful():
    model = build_review_model(**_review_kwargs(enrichment=None))
    assert model["enrichment_available"] is False
    blob = "\n".join(model["evidence"])
    assert "unavailable" in blob.lower()


def test_review_duplicate_evidence_only_when_real():
    dup = {"relationship_type": "DUPLICATE_CANDIDATE",
           "related_object": "s3://bucket/demo/other.txt"}
    with_dup = build_review_model(**_review_kwargs(relationships=[dup]))
    assert any("duplicate" in line.lower()
               for line in with_dup["evidence"])
    without_dup = build_review_model(**_review_kwargs())
    assert not any("duplicate" in line.lower()
                   for line in without_dup["evidence"])
    assert without_dup["relationships"] == []


def test_preview_content_is_bounded():
    text = "x" * 5000
    preview, total, truncated = preview_content(text, limit=2000)
    assert total == 5000
    assert truncated is True
    assert len(preview) == 2000


def test_preview_content_short_is_untruncated():
    preview, total, truncated = preview_content("hello", limit=2000)
    assert (preview, total, truncated) == ("hello", 5, False)


def test_keep_consequences_name_no_mutation():
    lines = action_consequences("KEEP", "demo/custom-doc.txt")
    blob = "\n".join(lines)
    assert "No S3 file move" in blob
    assert "semantic memory" in blob.lower()
    assert "demo/custom-doc.txt" in blob
    assert "trash/" not in blob


def test_quarantine_consequences_name_destination_and_verification():
    lines = action_consequences("QUARANTINE", "demo/custom-doc.txt")
    blob = "\n".join(lines)
    assert "trash/demo/custom-doc.txt" in blob
    assert "erify" in blob  # verify destination + verify removal
    assert "only after" in blob.lower()


def test_policy_counts_normalize_mixed_case():
    rows = [{"Policy": "review"}, {"Policy": "REVIEW"},
            {"Policy": "archive"}, {"Policy": "retain"},
            {"Policy": "weird-value"}]
    counts = policy_counts(rows)
    assert counts == {"REVIEW": 2, "ARCHIVE": 1, "KEEP": 1, "OTHER": 1}


def test_sensitivity_and_category_counts():
    rows = [{"Sensitivity": "confidential", "Category": "Financial Reporting"},
            {"Sensitivity": "internal", "Category": "Financial Reporting"},
            {"Sensitivity": "internal", "Category": "Logs & System Events"}]
    assert sensitivity_counts(rows) == {"confidential": 1, "internal": 2}
    assert category_counts(rows) == {"Financial Reporting": 2,
                                     "Logs & System Events": 1}


def test_chart_helpers_take_only_current_rows():
    import inspect
    for fn in (policy_counts, sensitivity_counts, category_counts):
        params = list(inspect.signature(fn).parameters)
        assert params == ["rows"], f"{fn.__name__} must not take history"


def test_review_helpers_have_no_hardcoded_demo_filenames():
    import pathlib
    text = pathlib.Path("src/sms_agent/ui_state.py").read_text(encoding="utf-8")
    for name in ("employee-contacts.txt", "financial-report.txt",
                 "old-project-log.txt", "project-plan.txt"):
        assert name not in text
    assert "last week" not in text.lower()
    assert "last month" not in text.lower()
