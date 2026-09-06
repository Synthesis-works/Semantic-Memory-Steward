import pytest
from datetime import datetime, timezone, timedelta
from sms_agent.models import FileMetadata, SemanticAnalysisResult
from sms_agent.importance import ImportanceScorer

@pytest.fixture
def base_metadata():
    return FileMetadata(
        key="docs/readme.md",
        filename="readme.md",
        extension=".md",
        size_bytes=1024,
        created_at=datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc),
        is_duplicate=False
    )

@pytest.fixture
def base_analysis():
    return SemanticAnalysisResult(
        key="docs/readme.md",
        category="general",
        sensitivity="public",
        importance_score=0.5,
        confidence=0.9,
        reasoning="general text",
        recommended_action="retain"
    )

@pytest.fixture
def reference_time():
    return datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)

def test_recent_important_project_document(base_metadata, base_analysis, reference_time):
    # Setup for highly important document
    base_metadata.key = "projects/alpha/project-plan.md"
    base_metadata.filename = "project-plan.md"
    base_analysis.category = "planning"
    base_analysis.sensitivity = "confidential"
    
    scorer = ImportanceScorer(reference_time=reference_time)
    score = scorer.score(base_metadata, base_analysis)
    
    assert score.score > 0.8
    assert score.factors.recency == 1.0 # 0 days old
    assert score.factors.document_importance == 0.9 # keyword "project-plan"
    assert score.factors.content_relevance == 0.9 # "planning" category
    assert score.factors.sensitivity == 0.8 # confidential
    assert score.factors.duplicate_penalty == 0.0
    assert "Recent" in score.explanation
    assert "highly indicative document type" in score.explanation
    assert "strong operational relevance" in score.explanation

def test_old_low_value_document(base_metadata, base_analysis, reference_time):
    # Setup for old log file
    base_metadata.created_at = reference_time - timedelta(days=365)
    base_metadata.key = "logs/debug.log"
    base_metadata.filename = "debug.log"
    base_metadata.extension = ".log"
    base_analysis.category = "debug"
    
    scorer = ImportanceScorer(reference_time=reference_time)
    score = scorer.score(base_metadata, base_analysis)
    
    assert score.score < 0.5
    assert score.factors.recency < 0.1 # Very old
    assert score.factors.document_importance == 0.3
    assert score.factors.content_relevance == 0.5
    assert "Older" in score.explanation

def test_sensitivity_can_drive_high_importance(base_metadata, base_analysis, reference_time):
    base_analysis.sensitivity = "restricted"
    scorer = ImportanceScorer(reference_time=reference_time)
    score1 = scorer.score(base_metadata, base_analysis)
    
    base_analysis.sensitivity = "public"
    score2 = scorer.score(base_metadata, base_analysis)
    
    assert score1.score > score2.score
    assert score1.factors.sensitivity == 1.0
    assert "high sensitivity" in score1.explanation

def test_duplicate_penalty(base_metadata, base_analysis, reference_time):
    scorer = ImportanceScorer(reference_time=reference_time)
    
    score1 = scorer.score(base_metadata, base_analysis)
    
    base_metadata.is_duplicate = True
    score2 = scorer.score(base_metadata, base_analysis)
    
    assert score2.score < score1.score
    assert score2.factors.duplicate_penalty == 0.4
    assert "(flagged as a duplicate candidate)" in score2.explanation
    
    # Ensure it doesn't drop below 0
    base_metadata.created_at = reference_time - timedelta(days=5000)
    score3 = scorer.score(base_metadata, base_analysis)
    assert score3.score == 0.0

def test_missing_metadata(base_metadata, reference_time):
    # Unanalyzed state (no SemanticAnalysisResult)
    scorer = ImportanceScorer(reference_time=reference_time)
    score = scorer.score(base_metadata, None)
    
    assert 0.0 <= score.score <= 1.0
    assert score.factors.content_relevance == 0.5
    assert score.factors.sensitivity == 0.2

def test_score_bounded():
    scorer = ImportanceScorer()
    
    meta = FileMetadata(
        key="x", size_bytes=0, created_at=datetime.now(timezone.utc)
    )
    # Give extremely high/low inputs implicitly through dates
    meta.created_at = datetime.now(timezone.utc) + timedelta(days=100) # Future date
    
    score = scorer.score(meta)
    assert 0.0 <= score.score <= 1.0
