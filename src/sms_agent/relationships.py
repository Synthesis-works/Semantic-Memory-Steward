"""
Relationship analysis for Semantic Memory Steward.

Preserves all existing deterministic evidence sources (hash, ETag, filename).
Optionally extends with semantic similarity via vector search as additive evidence.

Priority of evidence (strongest wins):
  1. Cryptographic content hash match  → DUPLICATE_CONFIRMED  (conf 0.99)
  2. Exact text match                  → DUPLICATE_CONFIRMED  (conf 0.95)
  3. ETag + size match                 → DUPLICATE_CANDIDATE  (conf 0.80-0.90)
  4. Filename similarity               → RELATED             (conf 0.40-0.60)
  5. High semantic similarity (≥0.85)  → RELATED             (conf = similarity)
     Semantic similarity is additive evidence — it never overrides
     a stronger cryptographic duplicate signal.
"""
import logging
import re
from typing import List, Optional

from .models import FileMetadata, RelationshipResult

log = logging.getLogger(__name__)

_SEMANTIC_SIMILARITY_THRESHOLD = 0.85


class RelationshipAnalyzer:
    """Identifies duplicate and related files deterministically + semantically."""

    def __init__(self, vector_store=None):
        """
        Args:
            vector_store: Optional VectorStore implementation.
                          When provided, semantic similarity is used as
                          additional (additive) relationship evidence.
        """
        self._vector_store = vector_store

    def analyze(
        self,
        target: FileMetadata,
        target_content: Optional[str],
        candidates: List[FileMetadata],
        query_vector: Optional[List] = None,
    ) -> List[RelationshipResult]:
        """
        Analyze a target file against all candidate files.

        Runs deterministic comparison for each candidate, then optionally
        appends semantic similarity matches from the vector store.

        Args:
            target:        Metadata for the document being analyzed.
            target_content: Retrieved text content (may be None).
            candidates:    All other documents in the bucket.
            query_vector:  Pre-computed embedding vector for `target`.
                           If None, semantic search is skipped.
        """
        # --- Deterministic pass ---
        results_by_key: dict = {}
        for cand in candidates:
            if cand.key == target.key:
                continue
            res = self.compare(target, target_content, cand, None)
            if res.relationship_type != "NONE":
                related_key = cand.s3_uri or cand.key
                results_by_key[related_key] = res

        # --- Semantic pass (additive only) ---
        if query_vector and self._vector_store:
            try:
                matches = self._vector_store.search(
                    query_vector,
                    top_k=10,
                    threshold=_SEMANTIC_SIMILARITY_THRESHOLD,
                )
            except Exception as exc:
                log.warning("Vector search failed; skipping semantic evidence: %s", exc)
                matches = []

            for match in matches:
                match_uri = match.metadata.get("s3_uri", match.vector_id)
                # Skip self-match — build target URI defensively
                bucket = getattr(target, "bucket", None) or ""
                target_uri = target.s3_uri or (
                    f"s3://{bucket}/{target.key}" if bucket else target.key
                )
                if match_uri == target_uri:
                    continue
                # Semantic evidence is additive — only add if not already a stronger match
                if match_uri in results_by_key:
                    existing = results_by_key[match_uri]
                    if existing.relationship_type in ("DUPLICATE_CONFIRMED", "DUPLICATE_CANDIDATE"):
                        # Stronger deterministic signal already recorded; enrich evidence only
                        if "high semantic similarity" not in existing.evidence:
                            existing.evidence.append(
                                f"high semantic similarity ({match.similarity_score:.2f})"
                            )
                        continue
                # New semantic relationship
                results_by_key[match_uri] = RelationshipResult(
                    relationship_type="RELATED",
                    confidence=round(match.similarity_score, 4),
                    related_object=match_uri,
                    evidence=[f"high semantic similarity ({match.similarity_score:.2f})"],
                )

        return list(results_by_key.values())

    def _normalize_filename(self, filename: str) -> str:
        """Strip extensions, common copy suffixes, and lower-case to find similar names."""
        if not filename:
            return ""
        name = filename.lower()
        for ext in [".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".xlsx", ".log"]:
            if name.endswith(ext):
                name = name[:-len(ext)]
        name = re.sub(r'(-copy|_copy|\(\d+\)|_v\d+|-v\d+)$', '', name)
        return name.strip()

    def compare(
        self,
        target: FileMetadata,
        target_content: Optional[str],
        candidate: FileMetadata,
        candidate_content: Optional[str],
    ) -> RelationshipResult:
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

        # 4. Content Size (strengthens existing candidate evidence)
        if target.size_bytes == candidate.size_bytes:
            if rel_type == "DUPLICATE_CANDIDATE":
                evidence.append("same content size")
                conf = 0.9

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
            evidence=evidence,
        )
