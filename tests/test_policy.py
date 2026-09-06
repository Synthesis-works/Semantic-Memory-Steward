import pytest
from sms_agent.policy import PolicyEngine
from tests.fixtures import (
    important_plan_meta, important_plan_analysis,
    sensitive_meta, sensitive_analysis,
    old_log_meta, old_log_analysis,
    ambiguous_meta, ambiguous_analysis,
    trash_meta, trash_analysis
)
from sms_agent.models import SemanticAnalysisResult

def test_important_file_keep():
    engine = PolicyEngine()
    decision = engine.evaluate(important_plan_analysis, important_plan_meta)
    
    assert decision.action == "KEEP"
    assert decision.risk == "MEDIUM"  # Internal + high importance bumps risk slightly
    assert "High importance score (0.9)" in decision.reasons[1]

def test_sensitive_file_keep():
    engine = PolicyEngine()
    decision = engine.evaluate(sensitive_analysis, sensitive_meta)
    
    assert decision.action == "KEEP"
    assert decision.risk == "HIGH"
    assert "restricted" in decision.reasons[0]

def test_old_log_archive():
    engine = PolicyEngine()
    decision = engine.evaluate(old_log_analysis, old_log_meta)
    
    assert decision.action == "ARCHIVE"
    assert decision.risk == "LOW"
    assert not decision.requires_human_approval # Archiving public logs can be automated if desired, or maybe we want approval. The rule says false currently.

def test_ambiguous_review():
    engine = PolicyEngine()
    decision = engine.evaluate(ambiguous_analysis, ambiguous_meta)
    
    assert decision.action == "REVIEW"
    assert decision.requires_human_approval is True
    assert "Low confidence" in decision.reasons[1]

def test_trash_requires_approval():
    engine = PolicyEngine()
    decision = engine.evaluate(trash_analysis, trash_meta)
    
    assert decision.action == "TRASH"
    assert decision.requires_human_approval is True

def test_high_sensitivity_cannot_be_trashed():
    engine = PolicyEngine()
    # Force a bad analysis that says TRASH but sensitivity is restricted
    bad_analysis = SemanticAnalysisResult(
        key="temp/junk.tmp",
        category="unknown",
        sensitivity="restricted",
        importance_score=0.1,
        confidence=0.9,
        reasoning="Should be trashed",
        recommended_action="delete"
    )
    
    decision = engine.evaluate(bad_analysis, trash_meta)
    
    # Engine should override the analysis and KEEP it due to restricted sensitivity risk level
    assert decision.action == "KEEP"
    assert decision.risk == "HIGH"
