import re
from typing import List, Optional
from .models import FileMetadata, RelationshipResult

class RelationshipAnalyzer:
    """Deterministically identifies duplicate and related files."""
    
    def analyze(self, target: FileMetadata, target_content: Optional[str], candidates: List[FileMetadata]) -> List[RelationshipResult]:
        """
        Analyze a target file against a list of candidates.
        Currently operates in O(N) since we compare one against many, avoiding N^2 bucket-wide checks.
        """
        results = []
        for cand in candidates:
            if cand.key == target.key:
                continue
                
            res = self.compare(target, target_content, cand, None) # We do not fetch candidate content just for relationships to save I/O
            if res.relationship_type != "NONE":
                results.append(res)
                
        return results
        
    def _normalize_filename(self, filename: str) -> str:
        """Strip extensions, common copy suffixes, and lower-case to find similar names."""
        if not filename:
            return ""
        name = filename.lower()
        # Remove common extensions
        for ext in [".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".xlsx", ".log"]:
            if name.endswith(ext):
                name = name[:-len(ext)]
                
        # Remove common copy variants like "-copy", "(1)", "_v2"
        name = re.sub(r'(-copy|_copy|\(\d+\)|_v\d+|-v\d+)$', '', name)
        return name.strip()

    def compare(self, target: FileMetadata, target_content: Optional[str], candidate: FileMetadata, candidate_content: Optional[str]) -> RelationshipResult:
        """Compare a target to a single candidate and return relationship evidence."""
        evidence = []
        rel_type = "NONE"
        conf = 0.0
        
        # 1. Cryptographic Content Hash
        if target.content_hash and candidate.content_hash and target.content_hash == candidate.content_hash:
            evidence.append("exact cryptographic content hash match")
            rel_type = "DUPLICATE_CONFIRMED"
            conf = 0.99
            
        # 2. Exact Text Match
        elif target_content is not None and candidate_content is not None:
            # Simple exact match (ignoring leading/trailing whitespace variations)
            if target_content.strip() == candidate_content.strip():
                evidence.append("exact normalized text match")
                rel_type = "DUPLICATE_CONFIRMED"
                conf = max(conf, 0.95)
                
        # 3. S3 ETag Match
        elif target.etag and candidate.etag and target.etag == candidate.etag:
            evidence.append("same S3 ETag")
            if rel_type == "NONE":
                rel_type = "DUPLICATE_CANDIDATE"
                conf = 0.8
                
        # 4. Content Size
        if target.size_bytes == candidate.size_bytes:
            # Only strengthens existing evidence, doesn't establish duplication on its own
            if rel_type == "DUPLICATE_CANDIDATE":
                evidence.append("same content size")
                conf = 0.9
            elif rel_type == "NONE":
                # Just size match, no other structural duplicate link
                pass
                
        # 5. Filename Similarity
        t_norm = self._normalize_filename(target.filename)
        c_norm = self._normalize_filename(candidate.filename)
        
        if t_norm and c_norm:
            if t_norm == c_norm:
                evidence.append("highly similar filename")
                if rel_type == "NONE":
                    rel_type = "RELATED"
                    conf = max(conf, 0.6)
            elif t_norm in c_norm or c_norm in t_norm:
                evidence.append("related filename")
                if rel_type == "NONE":
                    rel_type = "RELATED"
                    conf = max(conf, 0.4)
                    
        if rel_type == "NONE":
            conf = 0.0
            evidence = []
            
        return RelationshipResult(
            relationship_type=rel_type,
            confidence=conf,
            related_object=candidate.s3_uri or candidate.key,
            evidence=evidence
        )
