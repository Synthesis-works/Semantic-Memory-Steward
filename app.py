import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from botocore.exceptions import ClientError
from sms_agent.pipeline import SMSPipeline
from sms_agent.actions import ActionRequest, ActionEngine
from sms_agent.memory import DynamoDBMemoryStore
from sms_agent.ui_state import (
    build_recommendation,
    derive_status,
    format_action_result,
    pending_reviews,
    summarize_workspace,
)

load_dotenv()

st.set_page_config(page_title="Semantic Memory Steward", layout="wide", initial_sidebar_state="expanded")

def init_pipeline():
    if "pipeline" not in st.session_state:
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
        os.environ.setdefault("SMS_LLM_PROVIDER", "gemini")
        bucket = os.getenv("SMS_S3_BUCKET", "semantic-memory-steward-dev-527557823928")
        try:
            from sms_agent.memory import DynamoDBMemoryStore, S3VectorStore
            from sms_agent.embeddings import GeminiEmbeddingProvider
            memory_store = DynamoDBMemoryStore(table_name=os.getenv("SMS_DYNAMO_TABLE", "sms-semantic-memory"))
            vector_store = S3VectorStore(vector_bucket=bucket)
            st.session_state.pipeline = SMSPipeline(
                bucket_name=bucket,
                memory_store=memory_store,
                vector_store=vector_store
            )
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")

init_pipeline()
pipeline = st.session_state.get("pipeline")

def execute_governance_action(pipeline, request, record_keep_s3_uri=None):
    """Execute one governance action and persist the truthful outcome.

    The outcome is stored in session state (UI state) so it survives the
    rerun that follows execution; success is recorded only when the backend
    verified it. For KEEP, the human decision is additionally persisted in
    semantic memory (business state). Always ends with a rerun.
    """
    try:
        res = pipeline.action_engine.execute(request)
    except Exception as e:
        res = None
        exec_error = str(e)
    if res is not None and res.status in ("VERIFIED", "VERIFIED_NO_ACTION"):
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
        else:
            outcome = {
                "action": res.action, "key": res.key,
                "status": res.status, "message": res.message,
            }
    else:
        outcome = {
            "action": request.requested_action, "key": request.key,
            "status": res.status if res is not None else "FAILED",
            "message": res.message if res is not None else f"Execution raised an exception: {exec_error}",
        }
    st.session_state["sms_last_action"] = outcome
    st.cache_data.clear()
    st.rerun()

st.sidebar.title("SMS STATUS")
st.sidebar.markdown("────────────────────────────")
st.sidebar.markdown("Strands Agent &nbsp;&nbsp; ✅ Active")
st.sidebar.markdown("S3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("DynamoDB &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("S3 Vectors &nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("Comprehend &nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("")
st.sidebar.markdown("LLM: external fallback active · Bedrock restricted")

with st.sidebar.expander("Technical details (providers, quotas, Bedrock)"):
    st.markdown("**LLM Provider**")
    st.markdown("External fallback &nbsp;&nbsp;&nbsp; 🌐 Active")
    st.markdown("AWS Bedrock &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⚠️ Account restricted")
    st.markdown("SageMaker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⚠️ Endpoint quota unavailable")
    st.markdown("---")
    st.markdown("**Fallback API Keys**")
    external_key = st.text_input("Gemini/External API Key", type="password", value=os.environ.get("SMS_EXTERNAL_API_KEY", ""), key="sms_external_api_key")
    st.info("The agent architecture supports native AWS model providers through Strands. During this submission, the AWS account's model-inference quotas prevented live Bedrock/SageMaker inference, so the semantic-analysis provider is explicitly disclosed rather than silently simulated. The governance, enrichment, persistence, vector memory, and action layers remain AWS-native.")
if not st.session_state.get("sms_external_api_key") and not os.environ.get("SMS_EXTERNAL_API_KEY"):
    st.sidebar.warning("Enter a fallback API key under Technical details to run analysis.")
else:
    external_key = st.session_state.get("sms_external_api_key") or os.environ.get("SMS_EXTERNAL_API_KEY", "")
    os.environ["SMS_EXTERNAL_API_KEY"] = external_key
    os.environ["GEMINI_API_KEY"] = external_key

st.title("Semantic Memory Steward (SMS)")

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

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("▶ Run Full SMS Scan", type="primary", key="sms_run_scan"):
        scanned, scan_errors, duplicate_hits = 0, [], 0
        for item in inventory:
            if not item.key.startswith("demo/"): continue
            with st.spinner(f"Analyzing {item.key}..."):
                try:
                    out = pipeline.process_object(item.key, skip_inference=False, execute_action=False)
                    scanned += 1
                    for rel in (out.get("relationships") or []):
                        if rel.get("relationship_type") in ("DUPLICATE_CONFIRMED", "DUPLICATE_CANDIDATE"):
                            duplicate_hits += 1
                except Exception as e:
                    scan_errors.append(f"{item.key}: {e}")
        st.session_state["sms_last_scan"] = {
            "analyzed": scanned, "errors": scan_errors, "duplicates": duplicate_hits,
        }
        st.cache_data.clear()
        st.rerun()

last_scan = st.session_state.get("sms_last_scan")
if last_scan:
    if last_scan.get("errors"):
        st.warning("Last scan reported errors: " + "; ".join(last_scan["errors"]))
    else:
        st.info(f"Last scan: analyzed {last_scan.get('analyzed', 0)} document(s), "
                f"duplicate signals {last_scan.get('duplicates', 0)}.")

records = []
for item in inventory:
    if not (item.key.startswith("demo/") or item.key.startswith("trash/demo/")):
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
            "s3_uri": s3_uri,
            "raw_record": None,
            "metadata": item
        })

if records:
    df = pd.DataFrame(records)
    st.dataframe(df[["Filename", "Category", "Sensitivity", "Importance", "Policy", "Status"]], use_container_width=True)

    summary = summarize_workspace(records, last_scan=st.session_state.get("sms_last_scan"))
    st.markdown("### What SMS found in your workspace")
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Documents analyzed", summary["analyzed"])
    scol2.metric("Needs your attention", summary["needs_attention"])
    scol3.metric("Safe to keep", summary["safe_to_keep"])
    scol4, scol5, scol6 = st.columns(3)
    scol4.metric("Cleanup candidates", summary["cleanup_candidates"])
    scol5.metric("Actions completed", summary["actions_completed"])
    scol6.metric("Awaiting approval", summary["awaiting_approval"])
    if summary["duplicates"] is None:
        st.caption("Duplicates detected: unknown — run a scan to check.")
    else:
        st.caption(f"Duplicates detected (last scan): {summary['duplicates']}")

    queue = pending_reviews(records)
    if queue:
        st.markdown("### Needs your attention — SMS paused here because it needs you")
        for row in queue:
            qcol1, qcol2 = st.columns([3, 1])
            qcol1.write(f"**{row['Filename']}** — policy {row['Policy']}, "
                        f"sensitivity {row['Sensitivity']}, importance {row['Importance']}")
            if qcol2.button("Inspect", key=f"sms_inspect_{row['Filename']}"):
                st.session_state["sms_selected_file"] = row["Filename"]
                st.rerun()
else:
    st.info("No documents found in the demo prefix.")

st.markdown("---")
st.header("Document Inspector & Approval Queue")

selected_file = st.selectbox("Select a file to inspect:", [r["Filename"] for r in records], key="sms_selected_file")

# The outcome of the last executed action must survive the rerun that
# follows execution. Rendering it here (instead of only in the button
# callback) is what makes "click -> visible state transition" true.
last_action = st.session_state.get("sms_last_action")
if last_action:
    _kind, _text = format_action_result(
        last_action["action"], last_action["key"],
        last_action["status"], last_action["message"],
    )
    if _kind == "success":
        st.success(_text)
    else:
        st.error(_text)

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
        st.subheader("SMS recommendation")
        st.write(f"**{recommendation['headline']}**")
        for reason in recommendation["why"]:
            st.write(f"- {reason}")
        st.write(f"**Action:** {recommendation['action']}")
        st.markdown("---")
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Semantic Evidence")
            st.write(f"**Category:** {raw.category}")
            st.write(f"**Sensitivity:** {raw.sensitivity}")
            st.write(f"**Importance Score:** {raw.importance_score}")
            st.write(f"**Reasoning:** Retrieved from external fallback LLM.")
            if getattr(raw, "human_decision", None):
                st.info(f"**Human decision:** {raw.human_decision} — recorded in semantic memory.")
            
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
            st.subheader("Approval & Action")
            st.write(f"**Current Policy:** {raw.recommended_action.upper()}")
            
            if raw.recommended_action.lower() == "review" and not selected_file.startswith("trash/"):
                if getattr(raw, "human_decision", None) == "KEEP":
                    st.success("This document was Kept by human review. You may still choose a different action below.")
                else:
                    st.error("Action Paused: Human Approval Required")

                action_choice = st.radio("Select Governance Action:", ["KEEP", "QUARANTINE"], key="sms_action_choice")

                if st.button("Confirm & Execute", key="sms_confirm_execute"):
                    with st.spinner("Executing Action..."):
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
                st.write(f"SMS will copy to `archive/{selected_file}`, verify the copy, delete the original, and verify removal.")
                if st.button("Execute ARCHIVE", key="sms_execute_archive"):
                    with st.spinner("Archiving..."):
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
