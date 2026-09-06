import pytest
from datetime import datetime, timezone
from sms_agent.models import FileMetadata, RelationshipResult
from sms_agent.relationships import RelationshipAnalyzer

@pytest.fixture
def target_metadata():
    return FileMetadata(
        key="docs/report.txt",
        filename="report.txt",
        extension=".txt",
        size_bytes=100,
        created_at=datetime.now(timezone.utc),
        etag="\"hash123\"",
        content_hash="sha256_abc",
        is_duplicate=False
    )

@pytest.fixture
def candidate_metadata():
    return FileMetadata(
        key="docs/report-copy.txt",
        filename="report-copy.txt",
        extension=".txt",
        size_bytes=100,
        created_at=datetime.now(timezone.utc),
        etag="\"hash456\"",
        content_hash="sha256_def",
        is_duplicate=False
    )

@pytest.fixture
def analyzer():
    return RelationshipAnalyzer()

def test_exact_content_hash_match(analyzer, target_metadata, candidate_metadata):
    candidate_metadata.content_hash = "sha256_abc" # Same as target
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 1
    assert results[0].relationship_type == "DUPLICATE_CONFIRMED"
    assert results[0].confidence == 0.99
    assert "exact cryptographic content hash match" in results[0].evidence

def test_same_etag_candidate(analyzer, target_metadata, candidate_metadata):
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    candidate_metadata.etag = "\"hash123\""
    candidate_metadata.size_bytes = 200 # Diff size
    
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 1
    assert results[0].relationship_type == "DUPLICATE_CANDIDATE"
    assert results[0].confidence == 0.8
    assert "same S3 ETag" in results[0].evidence
    assert "same content size" not in results[0].evidence

def test_same_etag_and_size(analyzer, target_metadata, candidate_metadata):
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    candidate_metadata.etag = "\"hash123\""
    candidate_metadata.size_bytes = 100 # Same size
    
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 1
    assert results[0].relationship_type == "DUPLICATE_CANDIDATE"
    assert results[0].confidence == 0.9
    assert "same content size" in results[0].evidence

def test_similar_filename_supporting_evidence(analyzer, target_metadata, candidate_metadata):
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    target_metadata.etag = "a"
    candidate_metadata.etag = "b"
    target_metadata.size_bytes = 100
    candidate_metadata.size_bytes = 200
    
    # "report.txt" vs "report-copy.txt"
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 1
    assert results[0].relationship_type == "RELATED"
    assert results[0].confidence == 0.6
    assert "highly similar filename" in results[0].evidence

def test_same_size_alone_does_not_establish_duplication(analyzer, target_metadata, candidate_metadata):
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    target_metadata.etag = "a"
    candidate_metadata.etag = "b"
    candidate_metadata.filename = "unrelated.txt" # ensure filenames don't match
    candidate_metadata.key = "docs/unrelated.txt"
    target_metadata.size_bytes = 100
    candidate_metadata.size_bytes = 100
    
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 0 # NONE relationship is excluded from results

def test_exact_normalized_text_match(analyzer, target_metadata, candidate_metadata):
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    
    # Different everything else
    candidate_metadata.etag = "b"
    candidate_metadata.size_bytes = 999
    candidate_metadata.filename = "completely-different.txt"
    candidate_metadata.key = "diff.txt"
    
    results = analyzer.analyze(target_metadata, "Hello World \n", [candidate_metadata])
    # No candidate content provided -> NONE
    assert len(results) == 0
    
    # Let's test the `compare` directly with candidate content
    res = analyzer.compare(target_metadata, "Hello World \n", candidate_metadata, "Hello World")
    assert res.relationship_type == "DUPLICATE_CONFIRMED"
    assert res.confidence == 0.95
    assert "exact normalized text match" in res.evidence

def test_related_but_not_duplicate(analyzer, target_metadata, candidate_metadata):
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    target_metadata.filename = "project-plan.txt"
    candidate_metadata.filename = "project-plan-v2.txt"
    target_metadata.etag = "a"
    candidate_metadata.etag = "b"
    target_metadata.size_bytes = 100
    candidate_metadata.size_bytes = 200
    
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 1
    assert results[0].relationship_type == "RELATED"
    assert "highly similar filename" in results[0].evidence

def test_clearly_unrelated(analyzer, target_metadata, candidate_metadata):
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    target_metadata.filename = "alpha.txt"
    candidate_metadata.filename = "beta.jpg"
    target_metadata.etag = "a"
    candidate_metadata.etag = "b"
    target_metadata.size_bytes = 100
    candidate_metadata.size_bytes = 200
    
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 0

def test_missing_metadata_handled_safely(analyzer):
    meta1 = FileMetadata(key="1", size_bytes=10, created_at=datetime.now(timezone.utc))
    meta2 = FileMetadata(key="2", size_bytes=20, created_at=datetime.now(timezone.utc))
    results = analyzer.analyze(meta1, None, [meta2])
    assert len(results) == 0

def test_unsupported_large_content(analyzer, target_metadata, candidate_metadata):
    # Analyzer should not call out to S3. It only consumes provided content.
    target_metadata.content_hash = None
    candidate_metadata.content_hash = None
    candidate_metadata.etag = target_metadata.etag
    results = analyzer.analyze(target_metadata, None, [candidate_metadata])
    assert len(results) == 1
    assert results[0].relationship_type == "DUPLICATE_CANDIDATE" # Fired via ETag without text content
