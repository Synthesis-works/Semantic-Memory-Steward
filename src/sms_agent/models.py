from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator
from .economics import EconomicAssessment

class SemanticAnalysisResult(BaseModel):
    """Structured result of semantic analysis on a file."""
    
    key: str = Field(description="The unique identifier or path of the file (e.g., S3 key).")
    category: str = Field(description="Semantic category of the document (e.g., 'financial', 'hr', 'technical').")
    sensitivity: Literal["public", "internal", "confidential", "restricted"] = Field(
        description="The sensitivity level of the data."
    )
    importance_score: float = Field(
        ge=0.0, le=1.0, 
        description="A score from 0.0 to 1.0 indicating the document's business importance."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence score for this classification, from 0.0 to 1.0."
    )
    reasoning: str = Field(description="Short reasoning for the classification and scores.")
    recommended_action: Literal["retain", "archive", "review", "delete"] = Field(
        description="The recommended governance action. Destructive actions should require human approval."
    )
    economic_assessment: Optional["EconomicAssessment"] = Field(
        default=None,
        description=("Optional, purely informational economic-gate assessment "
                     "(labeled ESTIMATE). Layered ON TOP of the policy/safety/"
                     "approval decision: it can never override policy, authorize "
                     "an action, bypass human approval, or unlock mutations. "
                     "Back-compat: always defaults to None."),
    )

    @field_validator("recommended_action", mode="before")
    @classmethod
    def _normalize_recommended_action(cls, value):
        # LLMs and legacy records use varying case ("REVIEW", "Archive").
        # Normalize centrally so every construction site (Strands
        # structured output, manual fallback parsing, cache rebuild)
        # accepts case variants instead of crashing the scan. Unknown
        # values still fail validation loudly.
        if isinstance(value, str):
            return value.lower()
        return value

class FileMetadata(BaseModel):
    """File metadata from the storage system."""
    key: str
    filename: Optional[str] = None
    extension: Optional[str] = None
    size_bytes: int
    created_at: datetime
    last_modified_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    owner: Optional[str] = None
    content_hash: Optional[str] = None
    etag: Optional[str] = None
    s3_uri: Optional[str] = None
    bucket: Optional[str] = None
    is_duplicate: bool = False

class PolicyDecision(BaseModel):
    """The deterministic policy decision for a file."""
    action: Literal["KEEP", "ARCHIVE", "REVIEW", "TRASH"]
    requires_human_approval: bool
    risk: Literal["LOW", "MEDIUM", "HIGH"]
    reasons: List[str]

class RetrievedContent(BaseModel):
    """Domain model for retrieved object content."""
    bucket: str
    key: str
    content: str
    content_type: str
    size_bytes: int

class ImportanceFactors(BaseModel):
    """The individual factors contributing to the final importance score."""
    recency: float
    content_relevance: float
    sensitivity: float
    document_importance: float
    duplicate_penalty: float

class ImportanceScore(BaseModel):
    """The calculated deterministic importance of a document."""
    score: float = Field(ge=0.0, le=1.0)
    factors: ImportanceFactors
    explanation: str

class RelationshipResult(BaseModel):
    """Structured evidence of relationship between two files."""
    relationship_type: Literal["DUPLICATE_CONFIRMED", "DUPLICATE_CANDIDATE", "RELATED", "NONE"]
    confidence: float = Field(ge=0.0, le=1.0)
    related_object: str
    evidence: List[str]

from enum import Enum

class ExecutionMode(str, Enum):
    SAFE = "SAFE"
    AUTONOMOUS = "AUTONOMOUS"
    TURBO = "TURBO"

class ActionRequest(BaseModel):
    """A request to execute a specific action on an object."""
    s3_uri: str
    bucket: str
    key: str
    requested_action: Literal["KEEP", "ARCHIVE", "REVIEW", "QUARANTINE", "DELETE"]
    reason: str
    risk: Literal["LOW", "MEDIUM", "HIGH"]
    execution_mode: ExecutionMode = ExecutionMode.SAFE
    human_approved: bool = False

class ActionResult(BaseModel):
    """The result of an executed action."""
    action: Literal["KEEP", "ARCHIVE", "REVIEW", "QUARANTINE", "DELETE"]
    key: str
    status: Literal["VERIFIED", "VERIFIED_NO_ACTION", "PENDING_APPROVAL", "BLOCKED", "FAILED"]
    message: str

class SemanticMemoryRecord(BaseModel):
    """Persistent semantic memory metadata for a document."""
    s3_uri: str
    bucket: str
    key: str
    content_hash: Optional[str] = None
    etag: str
    size_bytes: int
    category: str
    sensitivity: str
    importance_score: float
    analysis_timestamp: datetime
    embedding_model: Optional[str] = None  # None = no embedding exists (retry target)
    vector_id: Optional[str] = None  # None = no vector in the store (retry target)
    recommended_action: str = "retain"  # Cached to faithfully reconstruct analysis on cache hits
    human_decision: Optional[str] = None  # Human review outcome, e.g. "KEEP"; policy output untouched

class Embedding(BaseModel):
    """Domain model for a vector embedding."""
    vector_id: str
    vector: List[float]
    metadata: dict = Field(default_factory=dict)

class SemanticMatch(BaseModel):
    """A matched document from a semantic vector search."""
    vector_id: str
    similarity_score: float
    metadata: dict
