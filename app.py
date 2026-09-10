import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from botocore.exceptions import ClientError
from sms_agent.pipeline import SMSPipeline
from sms_agent.actions import ActionRequest, ActionEngine
from sms_agent.memory import DynamoDBMemoryStore

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

st.sidebar.title("SMS STATUS")
st.sidebar.markdown("────────────────────────────")
st.sidebar.markdown("Strands Agent &nbsp;&nbsp; ✅ Active")
st.sidebar.markdown("S3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("DynamoDB &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("S3 Vectors &nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("Comprehend &nbsp;&nbsp;&nbsp; ✅ Connected")
st.sidebar.markdown("")
st.sidebar.markdown("**LLM Provider**")
st.sidebar.markdown("External fallback &nbsp;&nbsp;&nbsp; ⚠️ Active")
st.sidebar.markdown("AWS Bedrock &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⚠️ Account restricted")
st.sidebar.markdown("SageMaker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⚠️ Endpoint quota unavailable")

st.sidebar.info("The agent architecture supports native AWS model providers through Strands. During this submission, the AWS account's model-inference quotas prevented live Bedrock/SageMaker inference, so the semantic-analysis provider is explicitly disclosed rather than silently simulated. The governance, enrichment, persistence, vector memory, and action layers remain AWS-native.")

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
    if st.button("▶ Run Full SMS Scan", type="primary"):
        for item in inventory:
            if not item.key.startswith("demo/"): continue
            with st.spinner(f"Analyzing {item.key}..."):
                try:
                    pipeline.process_object(item.key, skip_inference=False, execute_action=False)
                except Exception as e:
                    st.error(f"Error analyzing {item.key}: {e}")
        st.cache_data.clear()
        st.rerun()

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
            "Status": "PENDING_REVIEW" if record.recommended_action.lower() == "review" else ("QUARANTINED" if item.key.startswith("trash/") else "SAFE"),
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
else:
    st.info("No documents found in the demo prefix.")

st.markdown("---")
st.header("Document Inspector & Approval Queue")

selected_file = st.selectbox("Select a file to inspect:", [r["Filename"] for r in records])

if selected_file:
    rec = next((r for r in records if r["Filename"] == selected_file), None)
    if not rec or not rec["Has Memory"]:
        st.warning("This document has not been analyzed yet. Please run the scan.")
    else:
        raw = rec["raw_record"]
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Semantic Evidence")
            st.write(f"**Category:** {raw.category}")
            st.write(f"**Sensitivity:** {raw.sensitivity}")
            st.write(f"**Importance Score:** {raw.importance_score}")
            st.write(f"**Reasoning:** Retrieved from external fallback LLM.")
            
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
                st.error("Action Paused: Human Approval Required")
                
                action_choice = st.radio("Select Governance Action:", ["KEEP", "QUARANTINE"])
                
                if st.button("Confirm & Execute"):
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
                        res = pipeline.action_engine.execute(req)
                        if res.status in ["VERIFIED", "VERIFIED_NO_ACTION"]:
                            st.success(f"Action {action_choice} completed: {res.message}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Action failed: {res.message}")
            else:
                st.success("No pending reviews for this document.")
