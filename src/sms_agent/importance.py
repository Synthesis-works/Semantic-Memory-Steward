import math
from datetime import datetime, timezone
from typing import Optional

from .models import FileMetadata, SemanticAnalysisResult, ImportanceScore, ImportanceFactors

class ImportanceScorer:
    """Deterministically calculates document importance based on multiple factors."""
    
    # Heuristic keywords indicating important document types
    IMPORTANT_KEYWORDS = {
        "project-plan", "project", "contract", "report", "proposal",
        "requirements", "specification", "spec", "policy", "financial",
        "invoice", "roadmap", "meeting", "design", "architecture"
    }

    # High-relevance semantic categories
    HIGH_RELEVANCE_CATEGORIES = {"financial", "legal", "hr", "strategy", "technical", "planning"}
    
    def __init__(self, reference_time: Optional[datetime] = None):
        """
        Initialize the scorer.
        
        Args:
            reference_time: Optional explicit time for deterministic testing of recency.
                            If None, uses datetime.now(timezone.utc) during calculation.
        """
        self.reference_time = reference_time
        
    def _calculate_recency(self, metadata: FileMetadata) -> float:
        """Calculate a recency score from 0.0 (old) to 1.0 (new)."""
        now = self.reference_time or datetime.now(timezone.utc)
        target_date = metadata.last_modified_at or metadata.created_at
        
        if not target_date:
            return 0.5  # Neutral if unknown
            
        delta_days = (now - target_date).days
        if delta_days < 0:
            delta_days = 0 # Future dates? Cap at 0 days old
            
        # Exponential decay: half-life of roughly 90 days.
        # e^(-lambda * t) where lambda = ln(2)/90 ≈ 0.0077
        # At 0 days -> 1.0, 90 days -> 0.5, 365 days -> ~0.06
        score = math.exp(-0.0077 * delta_days)
        return max(0.0, min(1.0, score))
        
    def _calculate_content_relevance(self, analysis: Optional[SemanticAnalysisResult]) -> float:
        """Evaluate semantic relevance."""
        if not analysis:
            return 0.5 # Neutral if un-analyzed
            
        base = 0.5
        if analysis.category.lower() in self.HIGH_RELEVANCE_CATEGORIES:
            base = 0.9
            
        return base
        
    def _calculate_sensitivity(self, analysis: Optional[SemanticAnalysisResult]) -> float:
        """Evaluate sensitivity impact on importance."""
        if not analysis:
            return 0.2
            
        sensitivity_weights = {
            "public": 0.1,
            "internal": 0.5,
            "confidential": 0.8,
            "restricted": 1.0
        }
        return sensitivity_weights.get(analysis.sensitivity.lower(), 0.2)
        
    def _calculate_document_importance(self, metadata: FileMetadata) -> float:
        """Evaluate importance based on file path/name/extension."""
        filename = (metadata.filename or "").lower()
        key = metadata.key.lower()
        
        # Check if any important keyword is in the filename or path
        for kw in self.IMPORTANT_KEYWORDS:
            if kw in filename or kw in key:
                return 0.9
                
        # Some generic fallbacks based on extensions
        ext = (metadata.extension or "").lower()
        if ext in {".csv", ".md", ".pdf", ".docx", ".xlsx"}:
            return 0.6
            
        return 0.3
        
    def score(self, metadata: FileMetadata, analysis: Optional[SemanticAnalysisResult] = None) -> ImportanceScore:
        """
        Calculate the total deterministic importance score.
        """
        f_recency = self._calculate_recency(metadata)
        f_relevance = self._calculate_content_relevance(analysis)
        f_sensitivity = self._calculate_sensitivity(analysis)
        f_doc = self._calculate_document_importance(metadata)
        f_dup_penalty = 0.4 if metadata.is_duplicate else 0.0
        
        # Weighted combination of positive factors
        # recency: 20%, relevance: 30%, sensitivity: 20%, doc_importance: 30%
        base_score = (
            (f_recency * 0.20) +
            (f_relevance * 0.30) +
            (f_sensitivity * 0.20) +
            (f_doc * 0.30)
        )
        
        # Apply duplicate penalty
        final_score = base_score - f_dup_penalty
        
        # Normalize strictly to [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))
        
        # Generate explanation deterministically
        explanation_parts = []
        if f_recency >= 0.8:
            explanation_parts.append("recent")
        elif f_recency <= 0.2:
            explanation_parts.append("older")
            
        if f_doc >= 0.8:
            explanation_parts.append("highly indicative document type")
        else:
            explanation_parts.append("document")
            
        if f_relevance >= 0.8:
            explanation_parts.append("with strong operational relevance")
            
        if f_sensitivity >= 0.8:
            explanation_parts.append("and high sensitivity")
            
        if f_dup_penalty > 0:
            explanation_parts.append("(flagged as a duplicate candidate)")
            
        explanation = " ".join(explanation_parts).strip().capitalize() + "."
        
        return ImportanceScore(
            score=round(final_score, 4),
            factors=ImportanceFactors(
                recency=round(f_recency, 4),
                content_relevance=round(f_relevance, 4),
                sensitivity=round(f_sensitivity, 4),
                document_importance=round(f_doc, 4),
                duplicate_penalty=round(f_dup_penalty, 4)
            ),
            explanation=explanation
        )
