from datetime import datetime, timezone
from typing import List, Optional
from .models import SemanticAnalysisResult, FileMetadata, PolicyDecision, RelationshipResult

class PolicyEngine:
    """Deterministic policy engine for Semantic Memory Steward."""
    
    def evaluate(self, analysis: SemanticAnalysisResult, metadata: FileMetadata, relationships: Optional[List[RelationshipResult]] = None) -> PolicyDecision:
        reasons: List[str] = []
        action = "KEEP"
        requires_approval = False
        
        # Determine if there is strong duplicate evidence
        is_duplicate = metadata.is_duplicate
        if relationships:
            for rel in relationships:
                if rel.relationship_type in ("DUPLICATE_CONFIRMED", "DUPLICATE_CANDIDATE"):
                    is_duplicate = True
                    break

        # 1. Determine Risk Level based primarily on sensitivity and importance
        if analysis.sensitivity == "restricted":
            risk = "HIGH"
            reasons.append("File is classified as restricted sensitivity.")
        elif analysis.sensitivity == "confidential":
            risk = "MEDIUM"
            reasons.append("File is classified as confidential.")
        elif analysis.sensitivity == "internal":
            risk = "LOW"
            reasons.append("File is classified as internal.")
        else:
            risk = "LOW"
            reasons.append("File is public/non-sensitive.")
            
        if analysis.importance_score >= 0.8:
            reasons.append(f"High importance score ({analysis.importance_score}).")
            
        # 2. Determine Action
        
        # Rule: High confidence required for automatic ARCHIVE/TRASH. Otherwise REVIEW.
        if analysis.confidence < 0.7:
            action = "REVIEW"
            reasons.append("Low confidence in semantic analysis; requires review.")
            requires_approval = True
            
        # Rule: Duplicates typically need review unless they are trivial
        elif is_duplicate:
            action = "REVIEW"
            reasons.append("File is flagged as a potential duplicate.")
            requires_approval = True
            
        # Rule: High risk/sensitivity needs human review to determine handling
        elif risk in ["HIGH", "MEDIUM"]:
            action = "REVIEW"
            reasons.append("File is sensitive and requires human review.")
            
        # Rule: High importance (but low risk) should be kept active
        elif analysis.importance_score >= 0.8:
            action = "KEEP"
            reasons.append("Retaining active status due to high importance.")
            
        # Rule: TRASH logic - only if very low importance and public
        elif analysis.importance_score <= 0.2 and analysis.sensitivity == "public":
            action = "TRASH"
            reasons.append("File has very low importance and no sensitivity. Recommended for trash.")
            requires_approval = True # Destructive action always requires approval
            
        # Rule: ARCHIVE logic - low importance, old/unaccessed
        elif analysis.importance_score < 0.5:
            # Check age if last_accessed_at is available
            now = datetime.now(timezone.utc)
            ref_date = metadata.last_accessed_at or metadata.created_at
            
            # Simple heuristic: if it's older than ~1 year (365 days)
            age_days = (now - ref_date).days if ref_date.tzinfo else (datetime.now() - ref_date).days
            if age_days > 365:
                action = "ARCHIVE"
                reasons.append(f"File is old ({age_days} days) and of lower importance.")
            else:
                action = "KEEP"
                reasons.append("File is recent enough to keep active.")
        else:
            # Default fallback
            action = "KEEP"
            reasons.append("Default retention policy applied.")
            
        # Ensure destructive or review actions always require human approval
        if action in ["TRASH", "REVIEW"]:
            requires_approval = True
            
        # If risk is HIGH, any state changes might require approval, but KEEP is safe.
        
        return PolicyDecision(
            action=action,
            requires_human_approval=requires_approval,
            risk=risk,
            reasons=reasons
        )
