"""
Focused hermetic integration tests for the economic-ESTIMATE wiring.

Proves the end-to-end chain:

    policy decision
        -> economic assessment
        -> SemanticAnalysisResult.economic_assessment
        -> SemanticMemoryRecord.economic_assessment
        -> memory persistence / reload
        -> process_object() result

Safety invariants: the economic assessment must NEVER feed back into the
policy/authorization/action path.

All external boundaries (DynamoDB, S3 Vectors, embedding APIs, LLM agent)
are mocked. No live AWS or model calls are ever made.
"""
import datetime
import hashlib
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from sms_agent.economics import (
    compute_economic_assessment,
    EconomicAssessment,
    DEFAULT_PROCESSING_USD_PER_FILE,
)
from sms_agent.models import (
    SemanticMemoryRecord,
    SemanticAnalysisResult,
    FileMetadata,
    PolicyDecision,
    ActionRequest,
)
from sms_agent.pipeline import SMSPipeline, _sha256
from sms_agent.memory import DynamoDBMemoryStore
from sms_agent.relationships import RelationshipAnalyzer

_GB = 1024 ** 3
_DT = datetime.datetime(2026, 9, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _assess(potential_saving_bytes):
    return compute_economic_assessment(
        potential_saving_bytes=potential_saving_bytes,
        processing_cost_usd=DEFAULT_PROCESSING_USD_PER_FILE,
    )


def _make_record(**kwargs) -> SemanticMemoryRecord:
    defaults = dict(
        s3_uri="s3://bucket/doc.txt",
        bucket="bucket",
        key="doc.txt",
        content_hash="abc123",
        etag='"etag-1"',
        size_bytes=500,
        category="Finance",
        sensitivity="internal",
        importance_score=0.4,
        analysis_timestamp=_DT,
        embedding_model="gemini-embedding-2",
        vector_id="vid001",
        recommended_action="archive",
        human_decision=None,
        economic_assessment=None,
    )
    defaults.update(kwargs)
    return SemanticMemoryRecord(**defaults)


def _make_pipeline(**overrides):
    """Minimal mock-heavy pipeline. Default: fresh document (no memory).

    set_policy: set pipeline.policy.evaluate to a fixed PolicyDecision.
    set_scorer: set pipeline.scorer.score to a fixed ImportanceScore.
    """
    mock_s3 = MagicMock()
    size_bytes = overrides.pop("size_bytes", 100 * 1024 * 1024 * 1024)  # 100 GB
    meta = FileMetadata(
        key="doc.txt", size_bytes=size_bytes, created_at=_DT,
        etag='"etag-1"', bucket="bucket",
        s3_uri="s3://bucket/doc.txt", filename="doc.txt",
    )
    mock_inventory = MagicMock()
    mock_inventory.collect.return_value = [meta]

    from sms_agent.models import RetrievedContent
    mock_reader = MagicMock()
    mock_reader.get_text.return_value = RetrievedContent(
        bucket="bucket", key="doc.txt",
        content="A monthly financial statement.",
        content_type="text/plain", size_bytes=size_bytes,
    )

    mock_agent = MagicMock()
    mock_agent.model_id = "test-model"
    mock_agent.analyze_file.return_value = SemanticAnalysisResult(
        key="doc.txt", category="Finance", sensitivity="internal",
        importance_score=0.4, confidence=0.9,
        reasoning="Test", recommended_action="archive",
    )

    mock_memory = MagicMock()
    mock_memory.get_record.return_value = None
    mock_vector = MagicMock()
    mock_vector.search.return_value = []
    mock_vector.get_vector.return_value = None

    mock_embed = MagicMock()
    mock_embed.model_id = "gemini-embedding-2"
    mock_embed.embed_text.return_value = [0.1, 0.2, 0.3]

    pipeline = SMSPipeline(
        bucket_name="bucket",
        s3_client=mock_s3,
        memory_store=mock_memory,
        vector_store=mock_vector,
        embedding_provider=mock_embed,
    )
    pipeline.inventory = mock_inventory
    pipeline.reader = mock_reader
    pipeline.agent = mock_agent
    pipeline.relationship_analyzer = RelationshipAnalyzer(vector_store=mock_vector)

    for key, value in overrides.items():
        setattr(pipeline, key, value)

    return pipeline, mock_agent, mock_memory, mock_vector, mock_embed


def _mock_decision(action="ARCHIVE", risk="LOW", approve=False,
                   size_bytes=100 * 1024 * 1024 * 1024):
    """Pipeline whose policy is mocked to a fixed PolicyDecision.

    Returns the same 5-tuple as _make_pipeline().
    """
    pipeline, _a, mock_memory, _v, _e = _make_pipeline(size_bytes=size_bytes)
    policy = MagicMock()
    policy.evaluate.return_value = PolicyDecision(
        action=action, requires_human_approval=approve, risk=risk,
        reasons=["test"],
    )
    pipeline.policy = policy
    return pipeline, _a, mock_memory, _v, _e


# ---------------------------------------------------------------------------
# End-to-end: policy -> economics -> result -> memory
# ---------------------------------------------------------------------------

def test_archive_eligible_document_gets_assessment():
    # 100 GB ARCHIVE-eligible candidate -> genuine, WORTHWHILE estimate.
    pipeline, _, memory, _, _ = _mock_decision(
        action="ARCHIVE", size_bytes=100 * 1024 * 1024 * 1024)
    result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)

    assert result["status"] == "ANALYZED"
    analysis = result["analysis"]
    assert analysis["recommended_action"] == "archive"
    assert analysis["economic_assessment"] is not None
    assert analysis["economic_assessment"]["status"] == "WORTHWHILE"

def test_keep_document_gets_no_saving_assessment():
    pipeline, _, memory, _, _ = _mock_decision(action="KEEP")
    result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)

    analysis = result["analysis"]
    # KEEP is not a storage-saving candidate: NOT_WORTHWHILE, never counted.
    assert result["decision"]["action"] == "KEEP"
    assert analysis["economic_assessment"]["status"] == "NOT_WORTHWHILE"
    # recommended_action is the model recommendation, unchanged by economics.
    assert analysis["recommended_action"] == "archive"



def test_assessment_computed_after_policy_decision():
    pipeline, _, _, _, _ = _mock_decision(action="ARCHIVE")
    calls = []

    class _Probe:
        def __init__(self, method):
            self._method = method

        def __call__(self, *args, **kwargs):
            calls.append(self._method)
            if self._method == "economics":
                return _assess(100 * _GB)
            return PolicyDecision(
                action="ARCHIVE", requires_human_approval=False, risk="LOW",
                reasons=["test"],
            )

    pipeline.policy.evaluate = _Probe("policy")
    with patch("sms_agent.pipeline.compute_economic_assessment",
               new=_Probe("economics")):
        pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
    assert calls == ["policy", "economics"], (
        f"economics must be computed AFTER policy; got order {calls}"
    )


def test_assessment_carried_into_semantic_memory_record():
    # 100 GB ARCHIVE candidate -> a genuine WORTHWHILE estimate ($2.30 @ 0.023).
    pipeline, _, memory, _, _ = _mock_decision(
        action="ARCHIVE", size_bytes=100 * _GB)
    pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
    assert memory.save_record.call_count == 1
    record = memory.save_record.call_args[0][0]
    assert record.economic_assessment is not None
    assert record.economic_assessment.status == "WORTHWHILE"
    assert record.economic_assessment.estimated_benefit_usd == pytest.approx(
        (100 * _GB / _GB) * 0.023
    )


def test_assessment_survives_memory_round_trip():
    resource, table = MagicMock(), MagicMock()
    store = DynamoDBMemoryStore(
        table_name="test-table", dynamodb_resource=resource)

    rec = _make_record(economic_assessment=_assess(100 * _GB))
    resource.Table.return_value = table

    store.save_record(rec)
    table.put_item.assert_called_once()
    item = table.put_item.call_args[1]["Item"]
    assert item.get("economic_assessment") is not None
    assert item["economic_assessment"]["status"] == "WORTHWHILE"

    table.get_item.return_value = {"Item": item}
    reloaded = store.get_record("s3://bucket/doc.txt")
    assert reloaded is not None
    assert reloaded.economic_assessment is not None
    assert reloaded.economic_assessment.status == "WORTHWHILE"
    assert reloaded.economic_assessment.estimated_net_benefit_usd == pytest.approx(
        rec.economic_assessment.estimated_net_benefit_usd
    )


def test_legacy_record_without_assessment_still_loads():
    resource, table = MagicMock(), MagicMock()
    table.get_item.return_value = {"Item": {
        "s3_uri": "s3://bucket/old.txt",
        "bucket": "bucket",
        "key": "old.txt",
        "etag": '"e1"',
        "size_bytes": 100,
        "category": "Finance",
        "sensitivity": "internal",
        "importance_score": Decimal("0.500000"),
        "analysis_timestamp": "2026-09-01T12:00:00+00:00",
        "embedding_model": "gemini-embedding-2",
        "vector_id": "vid-old",
        "recommended_action": "archive",
    }}
    resource.Table.return_value = table
    store = DynamoDBMemoryStore(
        table_name="test-table", dynamodb_resource=resource)
    rec = store.get_record("s3://bucket/old.txt")
    assert rec is not None
    assert rec.recommended_action == "archive"
    assert rec.economic_assessment is None


# ---------------------------------------------------------------------------
# Cache / idempotency
# ---------------------------------------------------------------------------

def test_cache_hit_reuses_analysis_without_extra_llm_call():
    pipeline, agent, memory, vector, embed = _make_pipeline()
    content_hash = _sha256("A monthly financial statement.")
    # Stored estimate is WORTHWHILE for 3 GB ($0.069). The pipeline metadata
    # defaults to 100 GB ($2.30): if economics were recomputed the benefit
    # would differ, so pinning the stored value proves genuine reuse.
    stored = _assess(3 * _GB)
    memory.get_record.return_value = _make_record(
        content_hash=content_hash,
        etag='"etag-1"',
        size_bytes=3 * _GB,
        embedding_model="gemini-embedding-2",
        economic_assessment=stored,
    )
    vector.get_vector.return_value = [0.5, 0.6, 0.7]

    result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)

    agent.analyze_file.assert_not_called()          # no extra LLM
    embed.embed_text.assert_not_called()
    assert result["memory"]["reused_analysis"] is True
    assert result["memory"]["reused_embedding"] is True
    assert result["analysis"]["economic_assessment"] is not None
    assert result["analysis"]["economic_assessment"]["status"] == "WORTHWHILE"
    assert result["analysis"]["economic_assessment"]["estimated_benefit_usd"] == \
        pytest.approx((3 * _GB / _GB) * 0.023)


def test_first_run_does_not_trigger_extra_inference_for_economics():
    pipeline, agent, memory, _, _ = _mock_decision(action="ARCHIVE")
    result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
    # Exactly one agent analysis for the first run; economics adds zero inference.
    assert agent.analyze_file.call_count == 1
    assert result["analysis"]["economic_assessment"] is not None


# ---------------------------------------------------------------------------
# Safety: economics never touches the authorization/safety path
# ---------------------------------------------------------------------------

def test_economics_does_not_change_policy_or_request():
    policy = MagicMock()
    decision = PolicyDecision(
        action="ARCHIVE", requires_human_approval=False, risk="LOW",
        reasons=["test"],
    )
    policy.evaluate.return_value = decision
    pipeline, agent, memory, _, _ = _make_pipeline()
    pipeline.policy = policy

    result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)

    # Policy decision appears verbatim.
    assert result["decision"]["action"] == "ARCHIVE"
    assert result["decision"]["risk"] == "LOW"
    assert result["decision"]["requires_human_approval"] is False

    # Action request is exactly what the policy produced.
    req = result["action_request"]
    assert req["requested_action"] == "ARCHIVE"
    assert req["risk"] == "LOW"
    assert req["human_approved"] is False

    # The assessment is present but purely informational alongside it.
    assert result["analysis"]["economic_assessment"]["status"] == "NOT_WORTHWHILE" or \
        result["analysis"]["economic_assessment"]["status"] == "WORTHWHILE"
    # recommended_action unchanged.
    assert result["analysis"]["recommended_action"] == "archive"


def test_economics_never_adds_approval_or_destructive_surface():
    import sms_agent.economics as econ
    exports = {n for n in dir(econ) if not n.startswith("_")}
    # Economics exposes only the pure function + model; no action/authorize/
    # approve/delete/execute surface.
    for bad in ("approve", "authorize", "delete", "action", "execute",
                "quarantine"):
        assert not any(bad in n.lower() for n in exports), (
            f"economics must not expose {bad!r} surface: {exports}"
        )


def test_assessment_input_excludes_keep_review_trash():
    for action in ("KEEP", "REVIEW", "TRASH"):
        pipeline, _, _, _, _ = _mock_decision(action=action)
        with patch("sms_agent.pipeline.compute_economic_assessment",
                   wraps=compute_economic_assessment) as patched:
            pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        args = patched.call_args
        assert args is not None, "compute_economic_assessment must be called"
        potential = args[1].get("potential_saving_bytes")
        # Non-ARCHIVE decisions must not be counted as storage savings.
        assert potential in (0, None), (
            f"{action} must not be counted as a storage saving; got {potential!r}"
        )