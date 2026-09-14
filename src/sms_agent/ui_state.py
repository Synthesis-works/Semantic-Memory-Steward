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

import os

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


_WORKSPACE_PREFIX_ENV = "SMS_WORKSPACE_PREFIX"
_DEFAULT_WORKSPACE_PREFIX = "demo/"
_MANAGED_PREFIXES = ("trash/", "archive/")


def workspace_prefix() -> str:
    """Configured active-workspace prefix (defaults to demo/)."""
    prefix = os.environ.get(_WORKSPACE_PREFIX_ENV, _DEFAULT_WORKSPACE_PREFIX)
    return prefix if prefix else _DEFAULT_WORKSPACE_PREFIX


def is_managed_document(key: str) -> bool:
    """Governance destinations (trash/, archive/) are never workspace docs."""
    return bool(key) and key.startswith(_MANAGED_PREFIXES)


def is_workspace_document(key: str) -> bool:
    """An active workspace document: under the workspace prefix and not
    a managed governance destination. No filenames are hardcoded."""
    if not key:
        return False
    if is_managed_document(key):
        return False
    return key.startswith(workspace_prefix())


_DOC_ACTIONS = ("KEEP", "QUARANTINE")


def get_doc_action(action_map: Optional[Dict[str, str]], key: str,
                   default: str = "KEEP") -> str:
    """Canonical selected action for one document (defaults to KEEP)."""
    action = (action_map or {}).get(key, default)
    return action if action in _DOC_ACTIONS else default


def set_doc_action(action_map: Dict[str, str], key: str,
                   action: str) -> str:
    """Record a human's selected action for one document."""
    if action not in _DOC_ACTIONS:
        raise ValueError(
            f"set_doc_action: unsupported action {action!r} "
            "(only KEEP/QUARANTINE are selectable)")
    action_map[key] = action
    return action


def result_for_doc(results: Optional[Dict[str, Any]],
                   key: str) -> Optional[Dict[str, Any]]:
    """Action outcome belonging to one document, or None."""
    return (results or {}).get(key)


def verified_destination(action: str, key: str) -> Optional[str]:
    """Destination the ActionEngine verifies moves to.

    Mirrors ActionEngine's derivation (trash/<key>, archive/<key>);
    KEEP moves nothing, so there is no destination.
    """
    if action == "QUARANTINE":
        return f"trash/{key}"
    if action == "ARCHIVE":
        return f"archive/{key}"
    return None


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


# Policy/sensitivity color mapping for consistent chart theming
_POLICY_COLORS = {
    "KEEP": "#059669",
    "RETAIN": "#059669",
    "SAFE": "#16a34a",
    "ARCHIVE": "#2563eb",
    "REVIEW": "#d97706",
    "QUARANTINE": "#dc2626",
    "TRASH": "#dc2626",
    "DELETE": "#dc2626",
    "OTHER": "#64748b",
}

_SENSITIVITY_COLORS = {
    "confidential": "#dc2626",
    "internal": "#d97706",
    "public": "#059669",
    "unknown": "#64748b",
}

_CATEGORY_COLORS = [
    "#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777",
    "#0891b2", "#ea580c", "#65a30d", "#9333ea", "#e11d48",
]


def _apply_theme_config(chart: alt.Chart, title: str, height: int) -> alt.Chart:
    """Apply consistent dark theme configuration to charts."""
    return chart.properties(
        title=alt.TitleParams(
            text=title,
            fontSize=13,
            fontWeight=600,
            color="#f1f5f9",
            subtitleColor="#94a3b8",
            subtitleFontSize=11,
        ),
        height=height,
    ).configure(
        axis=alt.AxisConfig(
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight=600,
            labelColor="#94a3b8",
            titleColor="#cbd5e1",
            gridColor="#334155",
            domainColor="#475569",
            tickColor="#475569",
        ),
        legend=alt.LegendConfig(
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight=600,
            labelColor="#cbd5e1",
            titleColor="#f1f5f9",
            orient="bottom",
        ),
        view=alt.ViewConfig(
            stroke="transparent",
        ),
        background="#0f172a",
    )


def render_donut(box, counts: Dict[str, int], title: str) -> bool:
    """Compact donut from real counts with center metric. Returns False when nothing to show."""
    items = [{"label": label, "value": value}
             for label, value in counts.items() if value > 0]
    if not items:
        return False
    df = pd.DataFrame(items)
    total = sum(item["value"] for item in items)
    # Determine color mapping based on label types (policy vs sensitivity)
    labels = [item["label"] for item in items]
    is_policy = any(l.upper() in _POLICY_COLORS for l in labels)
    is_sensitivity = any(l.lower() in _SENSITIVITY_COLORS for l in labels)
    if is_policy:
        color_scale = alt.Scale(
            domain=labels,
            range=[_POLICY_COLORS.get(l.upper(), "#64748b") for l in labels],
        )
    elif is_sensitivity:
        color_scale = alt.Scale(
            domain=labels,
            range=[_SENSITIVITY_COLORS.get(l.lower(), "#64748b") for l in labels],
        )
    else:
        color_scale = alt.Scale(domain=labels, range=_CATEGORY_COLORS)

    # Base donut chart
    donut = alt.Chart(df).mark_arc(innerRadius=60, stroke="#0f172a", strokeWidth=3).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("label:N", scale=color_scale, legend=alt.Legend(title=None, orient="bottom", labelFontSize=11, labelColor="#cbd5e1", titleColor="#f1f5f9")),
        tooltip=[alt.Tooltip("label:N", title=title), alt.Tooltip("value:Q", title="Documents", format=","), alt.Tooltip("pct:Q", title="%", format=".1f")],
    ).transform_calculate(pct="datum.value / " + str(total) + " * 100")

    # Center text with total count
    center_text = alt.Chart(pd.DataFrame({"total": [total], "title": [title]})).mark_text(
        align="center", baseline="middle", fontSize=24, fontWeight=700, color="#f1f5f9", dy=-8
    ).encode(text=alt.Text("total:Q", format=","))
    
    # Center subtitle
    center_sub = alt.Chart(pd.DataFrame({"title": [title]})).mark_text(
        align="center", baseline="middle", fontSize=11, color="#94a3b8", dy=18
    ).encode(text=alt.Text("title:N"))

    chart = alt.layer(donut, center_text, center_sub).resolve_scale(color="independent")
    chart = _apply_theme_config(chart, "", height=220)
    box.altair_chart(chart, use_container_width=True)
    return True


def render_category_bars(box, counts: Dict[str, int], title: str) -> bool:
    """Compact horizontal bars from real counts."""
    items = [{"label": label, "value": value}
             for label, value in counts.items() if value > 0]
    if not items:
        return False
    # Sort by value descending for readability
    items.sort(key=lambda x: x["value"], reverse=True)
    df = pd.DataFrame(items)
    # Assign consistent colors
    colors = [_CATEGORY_COLORS[i % len(_CATEGORY_COLORS)] for i in range(len(items))]
    color_scale = alt.Scale(domain=[item["label"] for item in items], range=colors)
    chart = alt.Chart(df).mark_bar(
        cornerRadiusTopRight=4, 
        cornerRadiusBottomRight=4,
        size=28,
    ).encode(
        x=alt.X("value:Q", title="Documents", axis=alt.Axis(grid=True, gridColor="#334155", gridDash=[2, 2])),
        y=alt.Y("label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=250, labelFontSize=11, labelColor="#cbd5e1", titleColor="#cbd5e1")),
        color=alt.Color("label:N", scale=color_scale, legend=None),
        tooltip=[alt.Tooltip("label:N", title="Category"), alt.Tooltip("value:Q", title="Documents", format=",")],
    )
    chart = _apply_theme_config(chart, title, height=max(180, 40 * len(items)))
    box.altair_chart(chart, use_container_width=True)
    return True


def render_impact_chart(box, points: List[Dict[str, Any]]) -> bool:
    """Line chart: unmanaged baseline vs SMS-managed storage (projected).

    Points come from impact.projection_points: every point is labeled
    projected because no historical measurements exist.
    """
    rows = []
    for point in points or []:
        rows.append({"month": point["month"], "series": "Without SMS",
                     "bytes": point["baseline_bytes"]})
        rows.append({"month": point["month"], "series": "With SMS",
                     "bytes": point["managed_bytes"]})
    if not rows:
        return False
    df = pd.DataFrame(rows)
    
    # Calculate the actual reduction for annotation
    baseline = points[0]["baseline_bytes"] if points else 0
    managed = points[0]["managed_bytes"] if points else 0
    reduction_bytes = baseline - managed
    reduction_pct = (reduction_bytes / baseline * 100) if baseline else 0
    
    # Format bytes for axis
    chart = alt.Chart(df).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=80), 
        strokeWidth=3
    ).encode(
        x=alt.X("month:O", title="Month (projected)", axis=alt.Axis(labelAngle=0, labelColor="#94a3b8", titleColor="#cbd5e1", gridColor="#334155", gridDash=[2, 2])),
        y=alt.Y("bytes:Q", title="Active storage (GB)", axis=alt.Axis(format=".2f", gridColor="#334155", gridDash=[2, 2], labelColor="#94a3b8", titleColor="#cbd5e1")),
        color=alt.Color("series:N",
                        scale=alt.Scale(domain=["Without SMS", "With SMS"], range=["#dc2626", "#059669"]),
                        legend=alt.Legend(title=None, orient="bottom", labelFontSize=12, labelColor="#cbd5e1")),
        tooltip=[
            alt.Tooltip("month:O", title="Month"),
            alt.Tooltip("series:N", title="Scenario"),
            alt.Tooltip("bytes:Q", title="Storage (GB)", format=".3f"),
        ],
    )
    
    # Add reduction annotation
    reduction_text = f"Potential reduction: {reduction_bytes / 1024:.1f} KB ({reduction_pct:.1f}%)"
    annotation = alt.Chart(pd.DataFrame({
        "month": [len(points) // 2] if points else [6],
        "bytes": [(baseline + managed) / 2] if points else [0],
        "text": [reduction_text]
    })).mark_text(
        align="center", baseline="middle", fontSize=12, fontWeight=600, color="#94a3b8", dy=-20
    ).encode(
        x=alt.X("month:O"),
        y=alt.Y("bytes:Q"),
        text=alt.Text("text:N"),
    )
    
    chart = (chart + annotation).resolve_scale(y="shared")
    chart = _apply_theme_config(chart, "Storage growth: unmanaged vs SMS-managed (Projected)", height=280)
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
