from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field

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
