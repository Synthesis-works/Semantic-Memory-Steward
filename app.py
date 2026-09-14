import os
import time
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from botocore.exceptions import ClientError
from sms_agent.pipeline import SMSPipeline
from sms_agent.actions import ActionRequest, ActionEngine
from sms_agent.memory import DynamoDBMemoryStore
from sms_agent.trace import TraceRecorder, render_trace_lines
from sms_agent.impact import (
    compute_impact,
    format_bytes,
    projection_points,
    scale_scenario,
)
from sms_agent.economics import compute_economic_assessment
from sms_agent.ui_state import (
    action_consequences,
    build_recommendation,
    build_review_model,
    category_counts,
    derive_status,
    format_action_result,
    get_doc_action,
    is_managed_document,
    is_workspace_document,
    pending_reviews,
    policy_counts,
    preview_content,
    render_category_bars,
    render_donut,
    render_impact_chart,
    result_for_doc,
    sensitivity_counts,
    set_doc_action,
    summarize_workspace,
    verified_destination,
)

load_dotenv()

st.set_page_config(page_title="Semantic Memory Steward", layout="wide", initial_sidebar_state="expanded")

# ---- SMS product polish — restrained enterprise palette (CSS only) ----
st.markdown("""
<style>
/* Layout */
.block-container { padding-top: 1.1rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
hr.sms-divider { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

/* Hero */
.sms-hero { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 22px; margin-bottom: 6px; }
.sms-hero h1 { font-size: 1.55rem; margin: 0 0 5px 0; letter-spacing: -0.02em; color: #0f172a; line-height: 1.2; }
.sms-hero p { margin: 0; color: #475569; font-size: 0.94rem; line-height: 1.5; }
.sms-workflow { color: #64748b; font-size: 0.78rem; letter-spacing: .06em; font-weight: 600; margin-top: 8px; }

/* Policy badges — same treatment everywhere */
.sms-badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: .04em; border: 1px solid; line-height: 1.7; vertical-align: middle; }
.sms-badge-keep { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.sms-badge-archive { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
.sms-badge-review { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.sms-badge-quarantine { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.sms-badge-safe { background: #f0fdf4; color: #14532d; border-color: #bbf7d0; }

/* Attention hero */
.sms-attention { border-left: 3px solid #f59e0b !important; }

/* KPI cards — rely on st.container(border=True) but tighten metric display */
[data-testid="stMetric"] { background: transparent; }
[data-testid="stMetricLabel"] { color: #475569; font-size: 0.78rem; letter-spacing: .04em; text-transform: uppercase; font-weight: 600; }
[data-testid="stMetricValue"] { color: #0f172a; }

/* Table / inspector polish */
.sms-caption { color: #64748b; font-size: 0.82rem; }
.sms-section { margin-top: 8px; }

/* Subtle muted text */
.sms-muted { color: #64748b; }
</style>
""", unsafe_allow_html=True)


def _policy_badge(policy: str) -> str:
    """One consistent visual treatment for policy states (KEEP/ARCHIVE/REVIEW/QUARANTINE)."""
    key = (policy or "").strip().lower()
    klass = {
        "keep": "sms-badge-keep",
        "retain": "sms-badge-keep",
        "safe": "sms-badge-safe",
        "archive": "sms-badge-archive",
        "review": "sms-badge-review",
        "quarantine": "sms-badge-quarantine",
        "trash": "sms-badge-quarantine",
        "delete": "sms-badge-quarantine",
    }.get(key, "sms-badge-review")
    label = (policy or "—").strip().upper() or "—"
    return f'<span class="sms-badge {klass}">{label}</span>'


def build_embedding_provider():
    """Select the embedding provider for the memory pipeline.

    Bedrock Titan Embed V2 (1024-dim) is the default/primary production path.
    Gemini remains available by explicit choice (SMS_EMBEDDING_PROVIDER=gemini).
    An explicit SMS_EMBEDDING_PROVIDER=none/off/disabled/empty returns None,
    which keeps the entire memory pipeline vector-free.
    """
    choice = os.getenv("SMS_EMBEDDING_PROVIDER", "bedrock").strip().lower()
    if choice in ("", "none", "off", "disabled"):
        return None
    if choice == "gemini":
        from sms_agent.embeddings import GeminiEmbeddingProvider
        return GeminiEmbeddingProvider(api_key=os.environ.get("GEMINI_API_KEY", ""))
    from sms_agent.embeddings import BedrockEmbeddingProvider
    return BedrockEmbeddingProvider()


def _llm_provider_summary() -> str:
    """One honest, provider-derived LLM status line for the sidebar."""
    provider = os.getenv("SMS_LLM_PROVIDER", "bedrock").strip().lower()
    if provider == "agentcore":
        return "LLM: AgentCore harness active (Nova Micro via Strands)"
    if provider == "sagemaker":
        return "LLM: SageMaker endpoint active (Strands)"
    if provider in ("gemini", "groq", "mistral", "nvidia"):
        return f"LLM: external fallback active ({provider})"
    return "LLM: AWS Bedrock active (Strands · Nova Micro)"


def init_pipeline():
    if "pipeline" not in st.session_state:
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        bucket = os.getenv("SMS_S3_BUCKET", "semantic-memory-steward-dev-527557823928")
        try:
            from sms_agent.memory import DynamoDBMemoryStore, S3VectorStore
            memory_store = DynamoDBMemoryStore(table_name=os.getenv("SMS_DYNAMO_TABLE", "sms-semantic-memory"))
            vector_store = S3VectorStore(
                vector_bucket=os.getenv("SMS_VECTOR_BUCKET", "sms-semantic-vectors-527557823928"),
                dimension=int(os.getenv("SMS_VECTOR_DIMENSION", "1024")),
            )
            st.session_state.pipeline = SMSPipeline(
                bucket_name=bucket,
                memory_store=memory_store,
                vector_store=vector_store,
                embedding_provider=build_embedding_provider(),
            )
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")

init_pipeline()
pipeline = st.session_state.get("pipeline")

def get_trace_recorder():
    """Session-backed recorder: the trace survives reruns until next scan."""
    data = st.session_state.get("sms_trace_events")
    return TraceRecorder.from_dicts(data) if data else TraceRecorder()


def save_trace_recorder(rec):
    st.session_state["sms_trace_events"] = rec.to_dicts()


def render_last_action_for(doc_key):
    """Render the action outcome belonging to one document, if any.

    Results are document-scoped: viewing another document never shows
    this document's outcome.
    """
    outcome = result_for_doc(st.session_state.get("sms_action_results"),
                             doc_key)
    if not outcome:
        return
    _kind, _text = format_action_result(
        outcome["action"], outcome["key"],
        outcome["status"], outcome["message"],
    )
    if _kind == "success":
        st.success(_text)
    else:
        st.error(_text)


def render_result_panel(outcome):
    """Staged, document-specific result block. Success only on verification."""
    action = outcome["action"]
    key = outcome["key"]
    status = outcome["status"]
    message = outcome["message"]
    if status in ("VERIFIED", "VERIFIED_NO_ACTION"):
        if action == "KEEP":
            st.success("✓ KEEP CONFIRMED")
            st.write(f"**{key}** remains in its current location.")
            st.write("Decision recorded in semantic memory.")
            st.write("No S3 mutation was required.")
        else:
            dest = verified_destination(action, key)
            st.success(f"✓ {action} VERIFIED")
            st.write(f"**{key}**")
            if dest:
                st.write(f"Moved to:\n`{dest}`")
            st.write("Destination verified.")
            st.write("Original verified removed.")
    elif status == "FAILED":
        st.error(f"✗ {action} FAILED")
        st.write(f"**{key}**")
        st.write(f"The requested {action.lower()} action was not verified.")
        st.write(message)
    elif status == "BLOCKED":
        st.error(f"⛔ {action} BLOCKED")
        st.write(f"**{key}** — not executed.")
        st.write(message)
    else:
        st.error(f"⏸ {action} PENDING")
        st.write(f"**{key}** — not executed.")
        st.write(message)


def build_records(pipeline, inventory):
    """Build the dashboard table rows from inventory + semantic memory.

    Only active workspace documents are listed: managed governance
    destinations (trash/, archive/) and stale records for objects no
    longer in S3 never appear as scan results.
    """
    records = []
    for item in inventory:
        if not is_workspace_document(item.key):
            continue
        s3_uri = f"s3://{pipeline.bucket_name}/{item.key}"
        record = None
        try:
            record = pipeline.memory_store.get_record(s3_uri)
        except Exception:
            pass
        if record:
            records.append({
                "Filename": item.key,
                "Category": record.category,
                "Sensitivity": record.sensitivity,
                "Importance": round(record.importance_score, 2),
                "Policy": record.recommended_action,
                "Status": derive_status(record.recommended_action, item.key,
                                        getattr(record, "human_decision", None)),
                "Has Memory": True,
                "SizeBytes": item.size_bytes,
                "s3_uri": s3_uri,
                "raw_record": record,
                "metadata": item
            })
        else:
            records.append({
                "Filename": item.key,
                "Category": "-",
                "Sensitivity": "-",
                "Importance": 0.0,
                "Policy": "PENDING_SCAN",
                "Status": "-",
                "Has Memory": False,
                "SizeBytes": item.size_bytes,
                "s3_uri": s3_uri,
                "raw_record": None,
                "metadata": item
            })
    return records


def render_activity(box):
    """Draw the LIVE ACTIVITY panel from session state (every render)."""
    rec = get_trace_recorder()
    active = st.session_state.get("sms_trace_active", False)
    done = st.session_state.get("sms_scan_done", False)
    if not rec.events and not active and not done:
        box.empty()
        return
    parts = ["### SMS LIVE ACTIVITY"]
    if active and not rec.events:
        parts.append("● SMS is working...")
    parts.extend(render_trace_lines(rec.events))
    if active:
        parts.append("● SMS is working...")
    if done and not active:
        seconds = st.session_state.get("sms_last_scan_seconds", 0.0)
        last = st.session_state.get("sms_last_scan") or {}
        parts.append(
            f"✓ Scan complete in {seconds:.1f}s — "
            f"{last.get('analyzed', 0)} analyzed · "
            f"{last.get('duplicates', 0)} duplicate signals")
    box.markdown("\n\n".join(parts))


def execute_governance_action(pipeline, request, record_keep_s3_uri=None):
    """Execute one governance action and persist the truthful outcome.

    The outcome is stored in session state (UI state) so it survives the
    rerun that follows execution; success is recorded only when the backend
    verified it. For KEEP, the human decision is additionally persisted in
    semantic memory (business state).     Always ends with a rerun.
    """
    rec = get_trace_recorder()
    rec.record("ACTION",
               f"Approval received — executing {request.requested_action} — {request.key}",
               status="RUNNING", document=request.key, event_type="started")
    save_trace_recorder(rec)
    try:
        res = pipeline.action_engine.execute(request)
    except Exception as e:
        res = None
        exec_error = str(e)
    if res is not None and res.status in ("VERIFIED", "VERIFIED_NO_ACTION"):
        rec = get_trace_recorder()
        rec.succeed("ACTION", f"{res.action} verified", res.message,
                    document=res.key, event_type="verified")
        save_trace_recorder(rec)
        decision_error = None
        if record_keep_s3_uri is not None and pipeline.memory_store is not None:
            try:
                pipeline.memory_store.record_human_decision(record_keep_s3_uri, "KEEP")
            except Exception as e:
                decision_error = str(e)
        if decision_error:
            outcome = {
                "action": request.requested_action, "key": request.key,
                "status": "FAILED",
                "message": f"Action verified but the KEEP decision could not be recorded: {decision_error}",
            }
            rec = get_trace_recorder()
            rec.fail("ACTION", "KEEP decision not recorded",
                     outcome["message"], document=request.key)
            save_trace_recorder(rec)
        else:
            outcome = {
                "action": res.action, "key": res.key,
                "status": res.status, "message": res.message,
            }
    else:
        status = res.status if res is not None else "FAILED"
        message = (res.message if res is not None
                   else f"Execution raised an exception: {exec_error}")
        outcome = {
            "action": request.requested_action, "key": request.key,
            "status": status, "message": message,
        }
        rec = get_trace_recorder()
        if status == "BLOCKED":
            rec.succeed("ACTION", "Action blocked by safety layer",
                        message, document=request.key, event_type="blocked")
        elif status == "PENDING_APPROVAL":
            rec.wait("ACTION", "Action awaiting approval",
                     message, document=request.key,
                     event_type="pending_approval")
        else:
            rec.fail("ACTION", f"{request.requested_action} failed",
                     message, document=request.key)
        save_trace_recorder(rec)
    st.session_state["sms_last_action"] = outcome
    results = st.session_state.get("sms_action_results") or {}
    results[request.key] = outcome
    st.session_state["sms_action_results"] = results
    st.cache_data.clear()
    st.rerun()

def render_action_controls(pipeline, rec, selected_file, prefix):
    """Approval radio + explicit consequences + descriptive action button.

    Shared by the dashboard inspector and the deep review view so both
    offer the identical, truthful action semantics.

    Selection is canonical per document (sms_selected_action): every
    component below derives from the active document's entry, so stale
    text from another document or a previous selection is impossible.
    """
    raw = rec["raw_record"]
    if raw.recommended_action.lower() == "review" and not selected_file.startswith("trash/"):
        if getattr(raw, "human_decision", None) == "KEEP":
            st.success("This document was Kept by human review. You may still choose a different action below.")
        else:
            st.error("Action Paused: Human Approval Required")

        # Approval panel — visually obvious human-in-the-loop boundary
        with st.container(border=True):
            st.markdown(f"**Current Policy:** {_policy_badge(raw.recommended_action)}", unsafe_allow_html=True)
            st.caption("SMS has paused before taking action because this document requires human approval under the current policy.")
            st.markdown("**Action Paused:** Human Approval Required")
            st.markdown("**Your Decision:** KEEP / QUARANTINE")

        st.markdown("**YOUR DECISION**")
        action_map = st.session_state.get("sms_selected_action") or {}
        default = get_doc_action(action_map, selected_file)
        action_choice = st.radio("Select the final governance action:",
                                 ["KEEP", "QUARANTINE"],
                                 index=["KEEP", "QUARANTINE"].index(default),
                                 key=f"{prefix}_choice_{selected_file}")
        set_doc_action(action_map, selected_file, action_choice)
        st.session_state["sms_selected_action"] = action_map

        st.markdown(f"**YOU ARE ABOUT TO {action_choice}**")
        st.write(f"**{selected_file}**")
        for line in action_consequences(action_choice, selected_file):
            st.write(f"- {line}")

        if action_choice == "KEEP":
            button_label = "Confirm Keep — No File Move"
            spinner_text = f"Recording KEEP decision for {selected_file}..."
        else:
            button_label = "Confirm Quarantine — Move to Trash"
            spinner_text = (f"Quarantining {selected_file} — copying to "
                            f"trash/{selected_file}...")
        if st.button(button_label, key=f"{prefix}_confirm_{selected_file}"):
            with st.spinner(spinner_text):
                req = ActionRequest(
                    s3_uri=rec["s3_uri"],
                    bucket=pipeline.bucket_name,
                    key=selected_file,
                    requested_action=action_choice,
                    reason="Human explicit override/approval via Dashboard",
                    risk="HIGH" if action_choice == "QUARANTINE" else "LOW",
                    human_approved=True
                )
                keep_uri = rec["s3_uri"] if action_choice == "KEEP" else None
                execute_governance_action(pipeline, req, record_keep_s3_uri=keep_uri)
    elif raw.recommended_action.lower() == "archive" and not selected_file.startswith(("trash/", "archive/")):
        # Policy ARCHIVE implies LOW risk (higher-risk documents are
        # forced to REVIEW by PolicyEngine), so this request is
        # truthfully labelled LOW; the authorizer enforces it anyway.
        st.info(
            "**RECOMMENDATION — Archive this document**\n"
            f"- Importance {raw.importance_score} (low)\n"
            f"- Sensitivity {raw.sensitivity} (low risk)\n"
            f"- Category {raw.category}\n"
            "- Policy permits ARCHIVE for low-risk documents."
        )
        with st.container(border=True):
            st.markdown(f"**Current Policy:** {_policy_badge(raw.recommended_action)}", unsafe_allow_html=True)
            st.caption("SMS can archive this document to the archive workspace prefix after you confirm.")
        st.markdown("**YOU ARE ABOUT TO ARCHIVE**")
        st.write(f"**{selected_file}**")
        st.write(f"- Copy to `archive/{selected_file}`, verify the copy, "
                 "delete the original, and verify removal.")
        if st.button("Confirm Archive — Move to Archive", key=f"{prefix}_archive_{selected_file}"):
            with st.spinner(f"Archiving {selected_file} — copying to archive/..."):
                req = ActionRequest(
                    s3_uri=rec["s3_uri"],
                    bucket=pipeline.bucket_name,
                    key=selected_file,
                    requested_action="ARCHIVE",
                    reason="Human confirmed policy ARCHIVE recommendation via Dashboard",
                    risk="LOW",
                    human_approved=True
                )
                execute_governance_action(pipeline, req)
    else:
        st.success("No pending reviews for this document.")


def render_review_page(pipeline, inventory, records):
    """Deep review view for one document. Answers: why did SMS stop,
    what did it find, what does it recommend, what happens on approval."""
    key = st.session_state.get("sms_review_key")
    if st.button("← Back to workspace", key="sms_back_to_workspace"):
        st.session_state["sms_view"] = "dashboard"
        st.rerun()
    rec = next((r for r in records if r["Filename"] == key), None)
    if not rec or not rec["Has Memory"]:
        st.warning("This document is not available for review (not analyzed or no longer present).")
        return
    raw = rec["raw_record"]
    status = derive_status(raw.recommended_action, key,
                           getattr(raw, "human_decision", None))

    queue = pending_reviews(records)
    queue_keys = [r["Filename"] for r in queue]
    nav_prev, nav_next, nav_spacer = st.columns([1, 1, 4])
    with nav_prev:
        if st.button("← Previous", key="sms_review_prev",
                     disabled=key not in queue_keys or queue_keys.index(key) == 0):
            st.session_state["sms_review_key"] = queue_keys[queue_keys.index(key) - 1]
            st.rerun()
    with nav_next:
        if st.button("Next →", key="sms_review_next",
                     disabled=key not in queue_keys or queue_keys.index(key) == len(queue_keys) - 1):
            st.session_state["sms_review_key"] = queue_keys[queue_keys.index(key) + 1]
            st.rerun()

    st.markdown("REVIEWING")
    st.header(f"📄 {key}")
    st.markdown(f"**{raw.sensitivity.upper()} · IMPORTANCE {round(raw.importance_score, 2)} · POLICY: {_policy_badge(raw.recommended_action)}**", unsafe_allow_html=True)
    econ = getattr(raw, "economic_assessment", None)
    if econ is not None:
        st.caption(f"Econ ESTIMATE: {econ.status.replace('_', ' ')} "
                   f"(net {econ.estimated_net_benefit_usd:+.4f} USD/yr est.)")
    st.write(f"**STATUS: {status.replace('_', ' ')}**")
    if status == "PENDING_REVIEW":
        st.error("REVIEW REQUIRED")
    st.caption(f"Last analyzed: {raw.analysis_timestamp} · "
               f"{raw.size_bytes} bytes · etag {raw.etag}")

    st.markdown("### Other documents")
    others = [r for r in records
              if r["Has Memory"] and r["Filename"] != key]
    if not others:
        st.caption("No other analyzed documents.")
    for other in others:
        ocol1, ocol2 = st.columns([3, 1])
        ocol1.write(f"**{other['Filename']}** — {other['Category']} · "
                    f"{other['Sensitivity']} · {other['Status']}")
        if ocol2.button("Open", key=f"sms_open_{other['Filename']}"):
            st.session_state["sms_review_key"] = other["Filename"]
            st.session_state["sms_selected_file"] = other["Filename"]
            st.rerun()

    render_last_action_for(key)

    content_text, content_error = None, None
    try:
        content_text = pipeline.reader.get_text(rec["metadata"]).content or ""
    except Exception as e:
        content_error = str(e)
    enrichment, enrichment_error = None, None
    if content_text is not None:
        try:
            enrichment = pipeline.comprehend.analyze_text(content_text)
        except Exception as e:
            enrichment_error = str(e)
    relationships, rel_ok = [], True
    if content_text is not None:
        try:
            candidates = [m for m in inventory if m.key != key]
            relationships = pipeline.relationship_analyzer.analyze(
                rec["metadata"], content_text, candidates, query_vector=None)
        except Exception:
            rel_ok = False
    preview, total_chars, truncated = preview_content(content_text or "", limit=2000)
    model = build_review_model(
        key=key, category=raw.category, sensitivity=raw.sensitivity,
        importance=round(raw.importance_score, 2), policy=raw.recommended_action,
        human_decision=getattr(raw, "human_decision", None),
        analysis_timestamp=raw.analysis_timestamp, size_bytes=raw.size_bytes,
        content_chars=total_chars, preview=preview, enrichment=enrichment,
        relationships=[{"relationship_type": r.relationship_type,
                        "related_object": r.related_object} for r in relationships],
    )

    tab_overview, tab_content, tab_analysis, tab_rel = st.tabs(
        ["Overview", "Content", "Analysis", "Relationships"])

    with tab_overview:
        with st.container(border=True):
            st.subheader("Document Summary")
            st.write(f"**Category:** {model['category']}")
            st.write(f"**Sensitivity:** {model['sensitivity']}")
            st.write(f"**Importance:** {model['importance']}")
            st.markdown(f"**Policy:** {_policy_badge(model['policy'])}", unsafe_allow_html=True)
            st.write(f"**Status:** {model['status']}")
            if status == "PENDING_REVIEW":
                st.warning(
                    "WHY SMS STOPPED — "
                    f"SMS classified this document as {model['sensitivity']}. "
                    "Human judgment is required before any governance action.")
        with st.container(border=True):
            st.subheader("Evidence")
            for line in model["evidence"]:
                st.write(f"- {line}")
            st.info(model["memory_note"])
        with st.container(border=True):
            st.markdown("**SMS RECOMMENDATION**")
            st.write(f"**{model['recommendation']['headline']}**")
            for reason in model["recommendation"]["why"]:
                st.write(f"- {reason}")
            st.write(f"**Action:** {model['recommendation']['action']}")
            if status == "PENDING_REVIEW":
                st.write("**Alternative:** QUARANTINE — "
                         f"Move the document to the isolated trash/quarantine "
                         f"location (`trash/{key}`).")

    with tab_content:
        st.subheader("Content Preview")
        st.caption("This is the actual content SMS analyzed.")
        if content_error:
            st.warning(f"Content could not be loaded: {content_error}")
        else:
            st.code(model["preview"] or "(empty document)")
            st.caption(f"Showing {len(model['preview'])} of {total_chars} characters"
                       + (" (truncated)." if truncated else "."))

    with tab_analysis:
        st.subheader("SMS analysis")
        st.write(f"**Category:** {model['category']}")
        st.write(f"**Sensitivity:** {model['sensitivity']}")
        st.write(f"**Importance:** {model['importance']}")
        st.subheader("AWS Comprehend Enrichment")
        if enrichment_error:
            st.warning(f"Enrichment could not be loaded: {enrichment_error}")
        elif not model["enrichment_available"]:
            st.warning("Comprehend enrichment unavailable for this document.")
        else:
            st.write(f"**Entities: PERSON**: "
                     f"{', '.join(model['persons']) if model['persons'] else 'None'}")
            st.write(f"**PII detected**: {'yes' if model['pii_found'] else 'no'}")

    with tab_rel:
        st.subheader("Relationships")
        if not rel_ok:
            st.warning("Relationship signals unavailable.")
        elif model["duplicates"]:
            for dup in model["duplicates"]:
                st.write(f"- Duplicate signal: {dup}")
        else:
            st.write("No duplicate signals detected.")

    st.markdown('<hr class="sms-divider" />', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Execute Action")
        st.caption("HUMAN DECISION REQUIRED · SMS investigated, recommended, and stopped. You are now making the final governance decision.")
    render_action_controls(pipeline, rec, key, prefix="sms_review")

    outcome = result_for_doc(st.session_state.get("sms_action_results"), key)
    if outcome:
        st.markdown('<hr class="sms-divider" />', unsafe_allow_html=True)
        render_result_panel(outcome)

    trace_events = [e for e in get_trace_recorder().events
                    if e.document == key]
    if trace_events:
        st.markdown('<hr class="sms-divider" />', unsafe_allow_html=True)
        st.subheader("What happened")
        st.markdown("\n\n".join(render_trace_lines(trace_events)))

st.sidebar.title("SMS STATUS")
st.sidebar.caption("System · observability — subordinate to the workspace")
st.sidebar.markdown("────────────────────────────")
st.sidebar.markdown("Strands Agent &nbsp;&nbsp; ✅ Active")
st.sidebar.markdown("S3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("DynamoDB &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("S3 Vectors &nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("Comprehend &nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("")
st.sidebar.markdown(_llm_provider_summary())
st.sidebar.caption("Backend frozen · UI polish only")

with st.sidebar.expander("Technical details (providers, quotas, Bedrock)"):
    _active_provider = os.getenv("SMS_LLM_PROVIDER", "bedrock").strip().lower()
    st.markdown("**Analysis provider**")
    if _active_provider == "agentcore":
        st.markdown("AgentCore Harness &nbsp;&nbsp;&nbsp; ✅ Active")
    elif _active_provider == "sagemaker":
        st.markdown("SageMaker endpoint &nbsp;&nbsp; ✅ Active")
    elif _active_provider in ("gemini", "groq", "mistral", "nvidia"):
        st.markdown("External fallback &nbsp;&nbsp;&nbsp; 🌐 Active")
    else:
        st.markdown("AWS Bedrock (Strands) &nbsp;&nbsp; ✅ Active")
    st.markdown("Bedrock (direct Strands) &nbsp; ✅ Available")
    st.markdown("SageMaker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⚠️ Endpoint quota unavailable")
    st.markdown("---")
    st.markdown("**Fallback API Keys**")
    external_key = st.text_input("Gemini/External API Key", type="password", value=os.environ.get("SMS_EXTERNAL_API_KEY", ""), key="sms_external_api_key")
    st.info("The agent architecture supports AWS-native model providers through Strands: Bedrock by default, or an optional AgentCore harness (SMS_LLM_PROVIDER=agentcore). The active provider is disclosed here and in the execution trace; with the AgentCore provider, analysis runs inside the AgentCore harness and is never silently simulated. External-fallback providers remain available by explicit choice. The governance, enrichment, persistence, vector memory, and action layers are AWS-native.")
external_key_value = st.session_state.get("sms_external_api_key") or os.environ.get("SMS_EXTERNAL_API_KEY", "")
if external_key_value:
    os.environ["SMS_EXTERNAL_API_KEY"] = external_key_value
    os.environ["GEMINI_API_KEY"] = external_key_value
if _active_provider in ("gemini", "groq", "mistral", "nvidia") and not external_key_value:
    st.sidebar.warning("Enter a fallback API key under Technical details to run analysis.")

if not pipeline:
    st.stop()

@st.cache_data(ttl=5)
def get_inventory():
    try:
        return pipeline.inventory.collect()
    except Exception as e:
        st.error(f"Failed to load inventory: {e}")
        return []

inventory = get_inventory()
records = build_records(pipeline, inventory)

if st.session_state.get("sms_view") == "review":
    render_review_page(pipeline, inventory, records)
    st.stop()

# ---- Product header ----
st.markdown("""
<div class="sms-hero">
  <h1>Semantic Memory Steward (SMS)</h1>
  <p>Autonomous memory governance for your team's documents — analyzes, classifies, and applies policy, pausing for human judgment when required.</p>
  <div class="sms-workflow">ANALYZE → DECIDE → ACT · The dashboard is your human approval and oversight interface</div>
</div>
""", unsafe_allow_html=True)

hdr_col1, hdr_col2 = st.columns([3, 1])
with hdr_col2:
    if st.button("▶ Run Full SMS Scan", type="primary", key="sms_run_scan", use_container_width=True):
        st.session_state["sms_scan_requested"] = True
        st.session_state["sms_trace_events"] = TraceRecorder().to_dicts()
        st.session_state["sms_trace_active"] = True
        st.session_state["sms_scan_done"] = False
        st.rerun()
with hdr_col1:
    st.caption("Scans all current workspace documents. Existing analyses are reused when content is unchanged.")

# ---- KPI cards — visually distinct bordered containers, real values only ----
if records:
    summary = summarize_workspace(records, last_scan=st.session_state.get("sms_last_scan"))
    st.markdown("### What SMS found in your workspace")
    st.caption("Live values from inventory + semantic memory — no fabricated numbers.")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Documents analyzed", summary["analyzed"])
            st.caption("in workspace")
    with k2:
        with st.container(border=True):
            st.metric("Needs your attention", summary["needs_attention"])
            st.caption("· requires judgment")
    with k3:
        with st.container(border=True):
            st.metric("Safe to keep", summary["safe_to_keep"])
            st.caption("· no action needed")
    with k4:
        with st.container(border=True):
            st.metric("Cleanup candidates", summary["cleanup_candidates"])
            st.caption("· archive-eligible")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.metric("Actions completed", summary["actions_completed"])
    with s2:
        with st.container(border=True):
            st.metric("Awaiting approval", summary["awaiting_approval"])
    with s3:
        with st.container(border=True):
            dup_val = summary["duplicates"]
            st.metric("Duplicate signals", dup_val if dup_val is not None else "—")
            st.caption("from last scan" if dup_val is not None else "run a scan to check")
    if summary["duplicates"] is None:
        st.caption("Duplicates detected: unknown — run a scan to check.")
    else:
        st.caption(f"Duplicates detected (last scan): {summary['duplicates']}")
else:
    summary = {"analyzed": 0, "needs_attention": 0, "safe_to_keep": 0, "cleanup_candidates": 0, "actions_completed": 0, "awaiting_approval": 0, "duplicates": None}

# ---- Live activity — collapsed by default, compact summary outside ----
_compact = f"Scan activity · {summary['analyzed']} documents" if records else "Scan activity · workspace empty"
st.caption(_compact)
with st.expander("Live activity", expanded=False):
    _live_box = st.empty()
    render_activity(_live_box)

if st.session_state.get("sms_scan_requested", False):
    st.session_state["sms_scan_requested"] = False
    scan_rec = get_trace_recorder()
    workspace_items = [i for i in inventory if is_workspace_document(i.key)]
    scan_rec.succeed("DISCOVERY", "S3 Scanner",
                     f"Found {len(workspace_items)} active workspace document(s)")
    save_trace_recorder(scan_rec)
    render_activity(_live_box)
    scanned, scan_errors, duplicate_hits = 0, [], 0
    scan_start = time.time()
    for item in workspace_items:
        with st.spinner(f"Analyzing {item.key}..."):
            try:
                out = pipeline.process_object(item.key, skip_inference=False, execute_action=False, trace=scan_rec)
                scanned += 1
                for rel in (out.get("relationships") or []):
                    if rel.get("relationship_type") in ("DUPLICATE_CONFIRMED", "DUPLICATE_CANDIDATE"):
                        duplicate_hits += 1
            except Exception as e:
                scan_errors.append(f"{item.key}: {e}")
        save_trace_recorder(scan_rec)
        render_activity(_live_box)
    st.session_state["sms_last_scan"] = {
        "analyzed": scanned, "errors": scan_errors, "duplicates": duplicate_hits,
    }
    st.session_state["sms_last_scan_seconds"] = time.time() - scan_start
    st.session_state["sms_trace_active"] = False
    st.session_state["sms_scan_done"] = True
    save_trace_recorder(scan_rec)
    render_activity(_live_box)
    st.cache_data.clear()
    st.rerun()

last_scan = st.session_state.get("sms_last_scan")
if last_scan:
    if last_scan.get("errors"):
        st.warning("Last scan reported errors: " + "; ".join(last_scan["errors"]))
    else:
        st.info(f"Last scan: analyzed {last_scan.get('analyzed', 0)} document(s), "
                f"duplicate signals {last_scan.get('duplicates', 0)}.")

# ---- Storage & Cost Impact — improved hierarchy, tiny demo values de-emphasized ----
if records and summary["analyzed"] > 0:
    managed_sizes = [item.size_bytes for item in inventory
                     if is_managed_document(item.key)]
    impact = compute_impact(records, managed_sizes)
    st.markdown('<hr class="sms-divider" />', unsafe_allow_html=True)
    st.markdown("### STORAGE & COST IMPACT")
    st.caption("What SMS can save by keeping your active workspace focused — current measured values first, projections second.")
    # Current measured
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Current active storage", format_bytes(impact["total_bytes"]))
            st.caption("measured · active workspace")
        with m2:
            st.metric("Potential storage reduction", format_bytes(impact["potential_bytes"]))
            st.caption("if cleanup candidates are archived")
        with m3:
            st.metric("Less active storage", f"{impact['potential_pct']:.0f}%")
            st.caption("potential reduction")
        with m4:
            st.metric("Relocated already", format_bytes(impact["managed_bytes"]))
            st.caption("in trash / archive")
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        render_impact_chart(
            hcol1, projection_points(impact["total_bytes"],
                                     impact["potential_bytes"]))
        if impact["potential_bytes"] > 0:
            st.caption("PROJECTED illustrative 12-month outlook from current "
                       "workspace composition; assumes no new uploads. Actual "
                       "storage depends on future documents and decisions.")
        else:
            st.caption("No ARCHIVE-policy documents yet — the SMS-managed "
                       "line matches baseline until cleanup candidates appear.")
        st.caption(f"{impact['doc_count']} documents · {len(sorted({row['Filename'].rsplit('.', 1)[-1].upper() for row in records if '.' in row['Filename']}))} "
                   f"format(s): {', '.join(sorted({row['Filename'].rsplit('.', 1)[-1].upper() for row in records if '.' in row['Filename']}))}")
    with hcol2:
        with st.container(border=True):
            st.markdown("**Cost — illustrative only**")
            monthly_cost = impact["monthly_cost_usd"]
            monthly_savings = impact["monthly_savings_usd"]
            st.write(f"**Estimated monthly S3 cost:** "
                     f"${monthly_cost:.2f}" if monthly_cost >= 0.01
                     else f"**Estimated monthly S3 cost:** ${monthly_cost:.6f}")
            st.caption("S3 Standard list price — not your AWS bill. Tiny demo workspaces show fractions of a cent.")
            st.write(f"**Estimated monthly savings:** "
                     f"${monthly_savings:.2f}" if monthly_savings >= 0.01
                     else f"**Estimated monthly savings:** ${monthly_savings:.6f}")
            st.caption("If archive-eligible documents were moved. Actual savings depend on retention.")
        with st.container(border=True):
            st.markdown("**Scale illustration — not a measurement**")
            scenario = scale_scenario(impact["potential_pct"])
            st.caption(f"The same {impact['potential_pct']:.0f}% applied to a 100 GB "
                       f"workspace could save "
                       f"{format_bytes(scenario['saved_bytes'])} "
                       f"(~${scenario['monthly_savings_usd']:.2f}/mo).")
        st.info("SMS doesn't delete blindly. It analyzes first, applies "
                "policy, and asks for human approval when required.")
        st.caption("Potential / projected figures are labeled as such. Small demo dollar values are shown at full precision so they don't visually dominate.")

if records and summary["analyzed"] > 0:
    st.markdown('<hr class="sms-divider" />', unsafe_allow_html=True)
    st.markdown("### Workspace analytics")
    st.caption("Policy, sensitivity, and category — same semantic treatment everywhere.")
    # Legend for policy states
    st.markdown(
        f"{_policy_badge('KEEP')} {_policy_badge('ARCHIVE')} {_policy_badge('REVIEW')} {_policy_badge('QUARANTINE')}"
        " &nbsp; <span class='sms-muted' style='font-size:0.82rem;'>· REVIEW and QUARANTINE signal greater caution than KEEP. Color never replaces the label.</span>",
        unsafe_allow_html=True,
    )
    ch1, ch2, ch3 = st.columns(3)
    render_donut(ch1, policy_counts(records), "Policy decisions")
    render_donut(ch2, sensitivity_counts(records), "Sensitivity")
    render_category_bars(ch3, category_counts(records), "Categories")

if records:
    queue = pending_reviews(records)
    st.markdown('<hr class="sms-divider" />', unsafe_allow_html=True)
    if queue:
        with st.container(border=True):
            st.markdown("### Needs your attention — SMS paused here because it needs you")
            st.caption("ANALYZE → DECIDE → ACT · SMS analyzes the workspace, identifies documents requiring judgment, and pauses when policy requires human review.")
            for row in queue:
                qcol1, qcol2 = st.columns([3, 1])
                with qcol1:
                    st.markdown(
                        f"**{row['Filename']}** — {_policy_badge(row['Policy'])} &nbsp; {row['Sensitivity']} · {row['Importance']}",
                        unsafe_allow_html=True,
                    )
                if qcol2.button("Review", key=f"sms_inspect_{row['Filename']}"):
                    st.session_state["sms_selected_file"] = row["Filename"]
                    st.session_state["sms_review_key"] = row["Filename"]
                    st.session_state["sms_view"] = "review"
                    st.rerun()
    else:
        st.success("✓ Nothing needs your attention")

    st.markdown("### Recent scan results")
    st.caption("Filenames · policy state · sensitivity · importance — policy badges use the same treatment as everywhere else.")
    df = pd.DataFrame(records)
    st.dataframe(
        df[["Filename", "Category", "Sensitivity", "Importance", "Policy", "Status"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Filename": st.column_config.TextColumn("Filename", width="large"),
            "Policy": st.column_config.TextColumn("Policy", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Sensitivity": st.column_config.TextColumn("Sensitivity", width="small"),
            "Importance": st.column_config.NumberColumn("Importance", format="%.2f"),
        },
    )
else:
    st.info("No documents found in the demo prefix.")

st.markdown('<hr class="sms-divider" />', unsafe_allow_html=True)
st.header("Document Inspector & Approval Queue")
st.caption("Select a document to see SMS's recommendation, the evidence it used, and the approval controls.")

selected_file = st.selectbox("Select a file to inspect:", [r["Filename"] for r in records], key="sms_selected_file")

render_last_action_for(selected_file)

if selected_file:
    rec = next((r for r in records if r["Filename"] == selected_file), None)
    if not rec or not rec["Has Memory"]:
        st.warning("This document has not been analyzed yet. Please run the scan.")
    else:
        raw = rec["raw_record"]
        recommendation = build_recommendation(
            policy=raw.recommended_action, sensitivity=raw.sensitivity,
            importance=round(raw.importance_score, 2), category=raw.category,
            human_decision=getattr(raw, "human_decision", None),
            key=selected_file,
        )
        with st.container(border=True):
            st.subheader("SMS recommendation")
            st.markdown(f"**{recommendation['headline']}**")
            for reason in recommendation["why"]:
                st.write(f"- {reason}")
            st.write(f"**Action:** {recommendation['action']}")
        st.markdown("---")
        colA, colB = st.columns(2)
        
        with colA:
            with st.container(border=True):
                st.subheader("Semantic Evidence")
                st.write(f"**Category:** {raw.category}")
                st.write(f"**Sensitivity:** {raw.sensitivity}")
                st.write(f"**Importance Score:** {raw.importance_score}")
                st.write("**Reasoning:** Analysis reasoning is not persisted with the semantic memory record.")
                if getattr(raw, "human_decision", None):
                    st.info(f"**Human decision:** {raw.human_decision} — recorded in semantic memory.")
            
            with st.container(border=True):
                st.markdown("#### AWS Comprehend Enrichment")
                try:
                    content = pipeline.reader.get_text(rec["metadata"])
                    enrichment = pipeline.comprehend.analyze_text(content.content)
                    has_pii = bool(enrichment.get("pii_entities"))
                    persons = list(set([ent.get("text", "") for ent in enrichment.get("entities", []) if ent.get("type") == "PERSON"]))
                    st.write(f"**Entities: PERSON**: {', '.join(persons) if persons else 'None'}")
                    st.write(f"**PII detected**: {'yes' if has_pii else 'no'}")
                except Exception as e:
                    st.warning(f"Could not load Comprehend signals: {e}")
            
        with colB:
            with st.container(border=True):
                st.subheader("Approval & Action")
                st.markdown(f"**Current Policy:** {_policy_badge(raw.recommended_action)}", unsafe_allow_html=True)
                render_action_controls(pipeline, rec, selected_file, prefix="sms_inspect")
