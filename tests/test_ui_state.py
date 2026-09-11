"""Unit tests for UI-state derivation helpers (sms_agent.ui_state).

These pure helpers are the testable seam for dashboard truthfulness:
- status shown in the document table must reflect real backend state,
- an action result may only be rendered as success when the backend
  actually verified it,
- workspace summaries and recommendations must be computed from real
  records, never hardcoded.
"""
from sms_agent.ui_state import (
    build_recommendation,
    derive_status,
    format_action_result,
    pending_reviews,
    summarize_workspace,
)


def test_review_without_decision_is_pending():
    assert derive_status("review", "demo/a.txt", None) == "PENDING_REVIEW"


def test_review_with_keep_decision_is_kept():
    assert derive_status("review", "demo/a.txt", "KEEP") == "KEPT"


def test_quarantined_prefix_wins():
    assert derive_status("review", "trash/demo/a.txt", "KEEP") == "QUARANTINED"


def test_non_review_is_safe():
    assert derive_status("retain", "demo/a.txt", None) == "SAFE"


def test_verified_keep_formats_as_success():
    kind, text = format_action_result("KEEP", "demo/a.txt", "VERIFIED_NO_ACTION",
                                      "No mutation required.")
    assert kind == "success"
    assert "KEEP" in text and "demo/a.txt" in text
    assert "verif" in text.lower()


def test_verified_mutation_formats_as_success():
    kind, text = format_action_result("QUARANTINE", "demo/a.txt", "VERIFIED",
                                      "Successfully quarantined.")
    assert kind == "success"
    assert "verif" in text.lower()


def test_failed_action_never_formats_as_success():
    kind, text = format_action_result("QUARANTINE", "demo/a.txt", "FAILED",
                                      "Source object does not exist.")
    assert kind == "error"
    assert "FAILED" in text
    assert "Source object does not exist." in text


def test_blocked_action_reports_not_executed():
    kind, text = format_action_result("DELETE", "demo/a.txt", "BLOCKED",
                                      "Action explicitly blocked by safety layer.")
    assert kind == "error"
    assert "not executed" in text.lower()


def test_pending_approval_reports_not_executed():
    kind, text = format_action_result("QUARANTINE", "demo/a.txt", "PENDING_APPROVAL",
                                      "Action requires explicit human approval.")
    assert kind == "error"
    assert "not executed" in text.lower()


def _row(key, policy, status, sensitivity="internal", importance=0.5):
    return {"Filename": key, "Policy": policy, "Status": status,
            "Sensitivity": sensitivity, "Importance": importance,
            "Has Memory": True}


def test_summarize_workspace_counts_from_rows():
    rows = [
        _row("demo/review.txt", "review", "PENDING_REVIEW",
             sensitivity="confidential", importance=0.8),
        _row("demo/kept.txt", "review", "KEPT"),
        _row("demo/old.txt", "archive", "SAFE", importance=0.2),
        _row("trash/demo/q.txt", "review", "QUARANTINED"),
    ]
    summary = summarize_workspace(rows, last_scan=None)
    assert summary["analyzed"] == 4
    assert summary["needs_attention"] == 1
    assert summary["safe_to_keep"] == 1
    assert summary["cleanup_candidates"] == 1
    assert summary["actions_completed"] == 2  # KEPT + QUARANTINED
    assert summary["awaiting_approval"] == 1
    assert summary["duplicates"] is None  # unknown until a scan runs


def test_summarize_workspace_uses_last_scan_duplicates():
    rows = [_row("demo/a.txt", "retain", "SAFE")]
    summary = summarize_workspace(rows, last_scan={"duplicates": 2, "analyzed": 1})
    assert summary["duplicates"] == 2


def test_pending_reviews_lists_only_undecided_review_rows():
    rows = [
        _row("demo/review.txt", "review", "PENDING_REVIEW"),
        _row("demo/kept.txt", "review", "KEPT"),
        _row("demo/old.txt", "archive", "SAFE"),
    ]
    queue = pending_reviews(rows)
    assert [r["Filename"] for r in queue] == ["demo/review.txt"]


def test_build_recommendation_review_uses_real_values():
    rec = build_recommendation(policy="review", sensitivity="confidential",
                               importance=0.83, category="hr",
                               human_decision=None, key="demo/x.txt")
    assert rec["headline"] == "REVIEW REQUIRED"
    assert any("confidential" in w for w in rec["why"])
    assert any("0.83" in w for w in rec["why"])
    assert "KEEP" in rec["action"] and "QUARANTINE" in rec["action"]


def test_build_recommendation_kept_states_no_action():
    rec = build_recommendation(policy="review", sensitivity="internal",
                               importance=0.9, category="technical",
                               human_decision="KEEP", key="demo/x.txt")
    assert "KEPT" in rec["headline"]
    assert "No further action" in rec["action"]


def test_build_recommendation_archive_names_destination():
    rec = build_recommendation(policy="archive", sensitivity="public",
                               importance=0.2, category="technical",
                               human_decision=None, key="demo/old.txt")
    assert "Archive" in rec["headline"]
    assert any("0.2" in w for w in rec["why"])
    assert "archive/demo/old.txt" in rec["action"]


def test_ui_state_module_has_no_hardcoded_demo_filenames():
    import pathlib
    text = pathlib.Path("src/sms_agent/ui_state.py").read_text(encoding="utf-8")
    for name in ("employee-contacts.txt", "financial-report.txt",
                 "old-project-log.txt", "project-plan.txt"):
        assert name not in text
