"""
Testable UI-state helpers for the SMS dashboard.

The dashboard must never claim success the backend did not verify, and the
document table must reflect real backend state (including recorded human
decisions). These pure helpers are the testable seam; app.py wires them to
Streamlit session state and rendering.
"""
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd

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


PREVIEW_LIMIT = 2000


def preview_content(text: str, limit: int = PREVIEW_LIMIT):
    """Bound a document preview. Returns (preview, total_chars, truncated)."""
    total = len(text or "")
    if total <= limit:
        return text or "", total, False
    return (text or "")[:limit], total, True


def action_consequences(action: str, key: str) -> List[str]:
    """Human-readable consequences of approving an action. Real values only."""
    if action == "QUARANTINE":
        return [
            f"Move {key} to the quarantine/trash location.",
            f"Copy to trash/{key}.",
            "Verify destination before touching the original.",
            "Remove original only after verification.",
            "Record the verified result.",
        ]
    return [
        f"Keep {key} in its current location.",
        "No S3 file move is required.",
        "The decision will be recorded in semantic memory.",
        "The review will be marked as resolved.",
    ]


_POLICY_LABELS = {
    "retain": "KEEP",
    "keep": "KEEP",
    "archive": "ARCHIVE",
    "review": "REVIEW",
    "delete": "DELETE",
    "trash": "TRASH",
    "quarantine": "QUARANTINE",
}


def policy_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count policies across current rows (mixed-case normalized)."""
    counts: Dict[str, int] = {}
    for row in rows:
        label = _POLICY_LABELS.get((row.get("Policy") or "").lower(), "OTHER")
        counts[label] = counts.get(label, 0) + 1
    return counts


def sensitivity_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count sensitivities across current rows."""
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get("Sensitivity") or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def category_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count categories across current rows."""
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get("Category") or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_review_model(
    key: str,
    category: Any,
    sensitivity: Any,
    importance: Any,
    policy: Optional[str],
    human_decision: Optional[str],
    analysis_timestamp: Any,
    size_bytes: Any,
    content_chars: int,
    preview: str,
    enrichment: Optional[Dict[str, Any]],
    relationships: List[Any],
) -> Dict[str, Any]:
    """Assemble everything the deep review view shows, from real values.

    No reasoning/confidence is included: the memory record does not store
    them, and they must not be invented. Duplicate evidence appears only
    when real relationship data is passed in.
    """
    status = derive_status(policy, key, human_decision)
    evidence = [
        f"Classified as {sensitivity}",
        f"Importance {importance} (category {category})",
    ]
    if (policy or "").lower() == "review":
        evidence.append("Policy requires human approval for this content.")
    if enrichment is None:
        evidence.append("AWS Comprehend enrichment unavailable.")
        enrichment_available = False
        persons: List[str] = []
        pii_found = False
    else:
        enrichment_available = True
        entities = enrichment.get("entities", []) or []
        persons = sorted({ent.get("text", "") for ent in entities
                          if ent.get("type") == "PERSON" and ent.get("text")})
        pii_found = bool(enrichment.get("pii_entities"))
        evidence.append(
            f"PERSON entities detected: "
            f"{', '.join(persons) if persons else 'None'}")
        evidence.append(f"PII: {'detected' if pii_found else 'not detected'}")
    dupes = []
    for rel in relationships or []:
        rel_type = (rel.get("relationship_type", "")
                    if isinstance(rel, dict) else
                    getattr(rel, "relationship_type", ""))
        if rel_type in ("DUPLICATE_CONFIRMED", "DUPLICATE_CANDIDATE"):
            target = (rel.get("related_object", "")
                      if isinstance(rel, dict) else
                      getattr(rel, "related_object", ""))
            dupes.append(target)
            evidence.append(f"Duplicate signal: {target}")
    if human_decision:
        memory_note = (f"Human decision {human_decision} recorded "
                       f"in semantic memory.")
    else:
        memory_note = ("SMS has retained this document's analysis "
                       "in semantic memory.")
    recommendation = build_recommendation(
        policy=policy, sensitivity=sensitivity, importance=importance,
        category=category, human_decision=human_decision, key=key)
    return {
        "key": key,
        "category": category,
        "sensitivity": sensitivity,
        "importance": importance,
        "policy": policy,
        "status": status,
        "human_decision": human_decision,
        "last_analyzed": str(analysis_timestamp),
        "size_bytes": size_bytes,
        "content_chars": content_chars,
        "preview": preview,
        "evidence": evidence,
        "memory_note": memory_note,
        "recommendation": recommendation,
        "enrichment_available": enrichment_available,
        "persons": persons,
        "pii_found": pii_found,
        "relationships": list(relationships or []),
        "duplicates": dupes,
    }


def render_donut(box, counts: Dict[str, int], title: str) -> bool:
    """Compact donut from real counts. Returns False when nothing to show."""
    items = [{"label": label, "value": value}
             for label, value in counts.items() if value > 0]
    if not items:
        return False
    df = pd.DataFrame(items)
    chart = alt.Chart(df).mark_arc(innerRadius=45).encode(
        theta="value",
        color=alt.Color("label", legend=alt.Legend(title=None)),
        tooltip=["label", "value"],
    ).properties(title=title, height=190)
    box.altair_chart(chart, use_container_width=True)
    return True


def render_category_bars(box, counts: Dict[str, int], title: str) -> bool:
    """Compact horizontal bars from real counts."""
    items = [{"label": label, "value": value}
             for label, value in counts.items() if value > 0]
    if not items:
        return False
    df = pd.DataFrame(items)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("value", title="documents"),
        y=alt.Y("label", title=None, sort="-x"),
        tooltip=["label", "value"],
    ).properties(title=title, height=max(120, 40 * len(items)))
    box.altair_chart(chart, use_container_width=True)
    return True

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
