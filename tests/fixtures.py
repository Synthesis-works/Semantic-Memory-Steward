from datetime import datetime, timedelta, timezone
from sms_agent.models import SemanticAnalysisResult, FileMetadata

now = datetime.now(timezone.utc)
one_year_ago = now - timedelta(days=366)

# 1. Important project plan (KEEP)
important_plan_meta = FileMetadata(
    key="projects/alpha/plan.docx",
    size_bytes=102400,
    created_at=now,
    last_accessed_at=now,
    owner="alice",
    is_duplicate=False
)
important_plan_analysis = SemanticAnalysisResult(
    key="projects/alpha/plan.docx",
    category="technical",
    sensitivity="internal",
    importance_score=0.9,
    confidence=0.95,
    reasoning="Core project plan.",
    recommended_action="retain"
)

# 2. Sensitive document (KEEP)
sensitive_meta = FileMetadata(
    key="hr/salary.xlsx",
    size_bytes=50000,
    created_at=now,
    owner="bob"
)
sensitive_analysis = SemanticAnalysisResult(
    key="hr/salary.xlsx",
    category="hr",
    sensitivity="restricted",
    importance_score=0.95,
    confidence=0.99,
    reasoning="Contains PII.",
    recommended_action="retain"
)

# 3. Old low-importance file (ARCHIVE)
old_log_meta = FileMetadata(
    key="logs/2024/app.log",
    size_bytes=5000000,
    created_at=one_year_ago,
    last_accessed_at=one_year_ago,
    owner="system"
)
old_log_analysis = SemanticAnalysisResult(
    key="logs/2024/app.log",
    category="technical",
    sensitivity="public",
    importance_score=0.3,
    confidence=0.9,
    reasoning="Old system log.",
    recommended_action="archive"
)

# 4. Ambiguous case (REVIEW)
ambiguous_meta = FileMetadata(
    key="unknown/notes.txt",
    size_bytes=100,
    created_at=now
)
ambiguous_analysis = SemanticAnalysisResult(
    key="unknown/notes.txt",
    category="unknown",
    sensitivity="internal",
    importance_score=0.5,
    confidence=0.4, # LOW CONFIDENCE
    reasoning="Could not determine content.",
    recommended_action="review"
)

# 5. Destructive TRASH recommendation
trash_meta = FileMetadata(
    key="temp/junk.tmp",
    size_bytes=10,
    created_at=one_year_ago,
    last_accessed_at=one_year_ago
)
trash_analysis = SemanticAnalysisResult(
    key="temp/junk.tmp",
    category="unknown",
    sensitivity="public",
    importance_score=0.05,
    confidence=0.95,
    reasoning="Empty temp file.",
    recommended_action="delete"
)
