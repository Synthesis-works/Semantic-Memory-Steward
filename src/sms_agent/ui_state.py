"""
Testable UI-state helpers for the SMS dashboard.

The dashboard must never claim success the backend did not verify, and the
document table must reflect real backend state (including recorded human
decisions). These pure helpers are the testable seam; app.py wires them to
Streamlit session state and rendering.
"""
from typing import Any, Dict, List, Optional, Tuple

#: ActionEngine statuses that mean "the backend verified the outcome".
SUCCESS_STATUSES = ("VERIFIED", "VERIFIED_NO_ACTION")


def derive_status(
    recommended_action: Optional[str],
    key: str,
    human_decision: Optional[str],
) -> str:
    """
    Derive the truthful table status for a document.

    A recorded KEEP decision resolves a REVIEW item to KEPT; an object that
    physically moved to the quarantine prefix is QUARANTINED regardless.
    """
    if key.startswith("trash/"):
        return "QUARANTINED"
    if (recommended_action or "").lower() == "review":
        return "KEPT" if human_decision == "KEEP" else "PENDING_REVIEW"
    return "SAFE"


def format_action_result(
    action: str,
    key: str,
    status: str,
    message: str,
) -> Tuple[str, str]:
    """
    Format an action outcome for display.

    Returns (kind, text) where kind is "success" ONLY when the backend
    verified the outcome. Non-executed outcomes (blocked / pending approval)
    say so explicitly; failures carry the backend reason.
    """
    if status in SUCCESS_STATUSES:
        return ("success", f"✅ {action} for {key} verified. {message}")
    if status == "PENDING_APPROVAL":
        return (
            "error",
            f"⏸️ {action} for {key} requires approval — not executed "
            f"({status}). {message}",
        )
    if status == "BLOCKED":
        return (
            "error",
            f"⛔ {action} for {key} blocked by safety layer — not executed "
            f"({status}). {message}",
        )
    return (
        "error",
        f"❌ {action} for {key} did not complete ({status}). {message}",
    )


def summarize_workspace(
    rows: List[Dict[str, Any]],
    last_scan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute the steward-style workspace summary from real table rows.

    Every count is derived from the rows passed in (which come from the
    inventory + semantic memory). Duplicate evidence is only available from
    an actual scan run, so without one the count is None (unknown) rather
    than a fabricated zero.
    """
    analyzed = [r for r in rows if r.get("Has Memory")]
    needs_attention = [r for r in analyzed if r.get("Status") == "PENDING_REVIEW"]
    kept = [r for r in analyzed if r.get("Status") == "KEPT"]
    safe_retained = [
        r for r in analyzed
        if r.get("Status") == "SAFE"
        and (r.get("Policy") or "").lower() != "archive"
    ]
    cleanup = [
        r for r in analyzed if (r.get("Policy") or "").lower() == "archive"
    ]
    completed = [
        r for r in analyzed if r.get("Status") in ("KEPT", "QUARANTINED")
    ]
    return {
        "analyzed": len(analyzed),
        "needs_attention": len(needs_attention),
        "safe_to_keep": len(kept) + len(safe_retained),
        "cleanup_candidates": len(cleanup),
        "actions_completed": len(completed),
        "awaiting_approval": len(needs_attention),
        "duplicates": (last_scan or {}).get("duplicates"),
    }


def pending_reviews(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows still waiting on a human decision."""
    return [r for r in rows if r.get("Status") == "PENDING_REVIEW"]


def build_recommendation(
    policy: Optional[str],
    sensitivity: Any,
    importance: Any,
    category: Any,
    human_decision: Optional[str],
    key: str,
) -> Dict[str, Any]:
    """
    Turn a policy decision + record values into a human-readable
    recommendation. Every "why" bullet cites an actual value — nothing is
    invented.
    """
    action_policy = (policy or "").lower()
    if action_policy == "review":
        if human_decision == "KEEP":
            return {
                "headline": "KEPT — human decision recorded",
                "why": ["You chose KEEP; no S3 changes were made."],
                "action": "No further action needed.",
            }
        return {
            "headline": "REVIEW REQUIRED",
            "why": [
                f"Sensitivity {sensitivity} — human judgment required by policy.",
                f"Importance {importance} (category {category}).",
            ],
            "action": "Choose KEEP or QUARANTINE below.",
        }
    if action_policy == "archive":
        return {
            "headline": "RECOMMENDATION — Archive this document",
            "why": [
                f"Importance {importance} (low — below retention threshold).",
                f"Sensitivity {sensitivity} (low risk).",
            ],
            "action": (
                f"SMS can archive this to archive/{key} "
                "(copy → verify → delete → verify) after you confirm."
            ),
        }
    return {
        "headline": "No action needed",
        "why": [
            f"Policy {policy}; sensitivity {sensitivity}; importance {importance}."
        ],
        "action": "None.",
    }
