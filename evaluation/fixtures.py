from dataclasses import dataclass
from typing import Optional

@dataclass
class SemanticExpectation:
    category: str
    sensitivity: str
    importance_tier: str  # e.g., "high", "low"

@dataclass
class GovernanceExpectation:
    recommended_action: str
    human_approval_required: bool
    autonomous_destruction_allowed: bool
    relationship_detected: bool = False

@dataclass
class EvalCase:
    name: str
    file_key: str
    semantic: SemanticExpectation
    governance: GovernanceExpectation
    mock_duplicate_s3_uri: Optional[str] = None


FIXTURES = [
    EvalCase(
        name="Project Plan",
        file_key="demo/project-plan.txt",
        semantic=SemanticExpectation(
            category="project",
            sensitivity="internal",
            importance_tier="high"
        ),
        governance=GovernanceExpectation(
            recommended_action="KEEP",
            human_approval_required=False,
            autonomous_destruction_allowed=False,  # generally important documents shouldn't be autonomously destroyed
        )
    ),
    EvalCase(
        name="Old Project Log",
        file_key="demo/old-project-log.txt",
        semantic=SemanticExpectation(
            category="history",
            sensitivity="internal",
            importance_tier="low"
        ),
        governance=GovernanceExpectation(
            recommended_action="ARCHIVE",
            human_approval_required=False,
            autonomous_destruction_allowed=True,
        )
    ),
    EvalCase(
        name="Employee Contacts",
        file_key="demo/employee-contacts.txt",
        semantic=SemanticExpectation(
            category="hr",
            sensitivity="confidential",
            importance_tier="high"
        ),
        governance=GovernanceExpectation(
            recommended_action="REVIEW",
            human_approval_required=True,
            autonomous_destruction_allowed=False,
        )
    ),
    EvalCase(
        name="Financial Report",
        file_key="demo/financial-report.txt",
        semantic=SemanticExpectation(
            category="finance",
            sensitivity="restricted",
            importance_tier="high"
        ),
        governance=GovernanceExpectation(
            recommended_action="REVIEW",
            human_approval_required=True,
            autonomous_destruction_allowed=False,
        )
    ),
    EvalCase(
        name="Duplicate Financial Report",
        file_key="demo/financial-report.txt", # Using the same content, will mock duplicate detection
        semantic=SemanticExpectation(
            category="finance",
            sensitivity="restricted",
            importance_tier="high"
        ),
        governance=GovernanceExpectation(
            recommended_action="REVIEW",
            human_approval_required=True,
            autonomous_destruction_allowed=False,
            relationship_detected=True
        ),
        mock_duplicate_s3_uri="s3://semantic-memory-steward-dev-527557823928/demo/duplicate-report.txt"
    )
]
