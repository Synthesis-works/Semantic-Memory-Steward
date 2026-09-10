"""
End-to-end processing pipeline for Semantic Memory Steward.

Integrates persistent semantic memory (DynamoDB + S3 Vectors) with
idempotency: documents are only re-analysed / re-embedded when their
content changes or the embedding model is upgraded.

The ActionEngine safety boundary is preserved in full.
execute_action=False means nothing in the memory layer mutates source S3
objects; persisting a SemanticMemoryRecord is purely an observation action.

Persistence ordering (HIGH-3):
  1. Vector is written to S3 Vectors first.
  2. Metadata is written to DynamoDB only if vector write succeeded.
  This guarantees DynamoDB never claims a vector_id that was not persisted.
  Orphaned vectors (vector stored, DynamoDB write failed) are possible but
  inert — they are simply re-written on the next analysis run.
"""
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .s3_inventory import S3InventoryCollector
from .s3_content import S3ContentReader
from .agent import SMSAgent
from .importance import ImportanceScorer
from .relationships import RelationshipAnalyzer
from .policy import PolicyEngine
from .models import (
    PolicyDecision, ActionRequest, SemanticMemoryRecord,
    Embedding, SemanticAnalysisResult,
)
from .actions import ActionEngine

log = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_vector_id(bucket: str, key: str) -> str:
    """Stable, URL-safe vector ID derived from bucket + key."""
    return hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()[:32]


def _content_changed(
    existing: Optional[SemanticMemoryRecord],
    new_content_hash: Optional[str],
    new_etag: Optional[str],
    new_size: int,
) -> bool:
    """
    Returns True when we should re-analyze/re-embed the document.

    Strong identity: content_hash (SHA-256 of retrieved text).
    Supporting evidence: etag + size_bytes.
    We do NOT treat ETag as cryptographic content identity.
    """
    if existing is None:
        return True  # Never seen before

    # Strong identity wins
    if new_content_hash and existing.content_hash:
        return new_content_hash != existing.content_hash

    # Fallback: ETag + size as supporting change signal
    if new_etag and existing.etag:
        return new_etag != existing.etag or new_size != existing.size_bytes

    # Cannot determine — treat as changed to be safe
    return True


def _embedding_stale(
    existing: Optional[SemanticMemoryRecord],
    current_model_id: str,
) -> bool:
    """True when the model changed (embeddings from different spaces)."""
    if existing is None:
        return True
    return existing.embedding_model != current_model_id


class SMSPipeline:
    """End-to-end processing pipeline for Semantic Memory Steward."""

    def __init__(
        self,
        bucket_name: str,
        s3_client=None,
        memory_store=None,         # SemanticMemoryStore | None
        vector_store=None,         # VectorStore | None
        embedding_provider=None,   # EmbeddingProvider | None
    ):
        self.bucket_name = bucket_name
        self.s3_client = s3_client

        # Domain layers
        from .comprehend import ComprehendAnalyzer
        self.inventory = S3InventoryCollector(bucket_name, s3_client=s3_client)
        self.reader = S3ContentReader(s3_client=s3_client)
        self.agent = SMSAgent()
        self.scorer = ImportanceScorer()
        self.relationship_analyzer = RelationshipAnalyzer(vector_store=vector_store)
        self.policy = PolicyEngine()
        self.action_engine = ActionEngine(s3_client=s3_client)
        self.comprehend = ComprehendAnalyzer()

        # Memory layer (optional — may be None when AWS resources aren't provisioned)
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _s3_uri(self, key: str) -> str:
        return f"s3://{self.bucket_name}/{key}"

    def _try_load_memory(self, s3_uri: str) -> Optional[SemanticMemoryRecord]:
        if self.memory_store is None:
            return None
        try:
            return self.memory_store.get_record(s3_uri)
        except Exception as exc:
            log.warning(
                "Memory store get_record failed (%s): %s — treating as new document",
                s3_uri, exc,
            )
            return None

    def _try_persist(
        self,
        record: SemanticMemoryRecord,
        embedding: Optional[Embedding],
    ) -> bool:
        """
        Persist vector first, then metadata.

        HIGH-3 ordering contract:
          - If no vector store is configured, skip vector step.
          - If vector upsert fails, do NOT write DynamoDB metadata.
            This prevents DynamoDB claiming a vector_id that was never stored.
          - If vector upsert succeeds but DynamoDB write fails, the vector
            is an orphan. This is acceptable for MVP — re-analysis will
            overwrite it on the next run.

        Returns True if both steps succeeded (or were intentionally skipped),
        False if any step failed.
        """
        # Step 1: Persist vector (if configured and embedding is available)
        if self.vector_store is not None and embedding is not None:
            try:
                self.vector_store.upsert(embedding)
            except Exception as exc:
                log.error(
                    "Vector upsert failed (vector_id=%s): %s — "
                    "DynamoDB metadata will NOT be written to prevent inconsistent state.",
                    embedding.vector_id, exc,
                )
                return False

        # Step 2: Persist metadata (only if vector step did not fail)
        if self.memory_store is not None:
            try:
                self.memory_store.save_record(record)
            except Exception as exc:
                log.error(
                    "DynamoDB save_record failed (%s): %s — "
                    "vector may be orphaned but DynamoDB state is not corrupted.",
                    record.s3_uri, exc,
                )
                return False

        return True

    def _try_get_cached_vector(self, vector_id: str) -> Optional[list]:
        """
        Retrieve a stored embedding vector for cache-hit documents.

        HIGH-2: When content is unchanged and embedding model is unchanged,
        we must NOT pass query_vector=None to the relationship analyzer, or
        semantic relationship search would be silently skipped.
        We retrieve the existing vector from S3 Vectors instead.
        """
        if self.vector_store is None:
            return None
        if not hasattr(self.vector_store, "get_vector"):
            return None
        try:
            return self.vector_store.get_vector(vector_id)
        except Exception as exc:
            log.warning("Could not retrieve cached vector for %s: %s", vector_id, exc)
            return None

    def _try_embed(self, text: str) -> Optional[list]:
        if self.embedding_provider is None:
            return None
        try:
            return self.embedding_provider.embed_text(text)
        except Exception as exc:
            log.warning("Embedding generation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_object(
        self,
        file_key: str,
        skip_inference: bool = True,
        human_approved: bool = False,
        execute_action: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single S3 object through the full pipeline.

        Args:
            file_key:       S3 object key.
            skip_inference: If True, halt before LLM semantic analysis.
            human_approved: Explicit human authorisation for mutations.
            execute_action: If False, compute recommendations but never
                            invoke ActionEngine mutations.

        Memory behaviour:
            - First encounter: full analysis + embed + persist (vector first,
              then metadata — HIGH-3 ordering).
            - Unchanged content + same embedding model: reuse cached analysis
              and retrieve existing vector for semantic search (HIGH-2).
            - Unchanged content + changed embedding model: reuse cached LLM
              analysis, generate fresh embedding only.
            - Changed content: full re-analysis + re-embed.
        """
        # 1. Inventory
        items = self.inventory.collect()
        metadata = next((m for m in items if m.key == file_key), None)
        if not metadata:
            raise ValueError(
                f"Object metadata not found in bucket {self.bucket_name} for key: {file_key}"
            )

        # 2. Content retrieval
        content = self.reader.get_text(metadata)

        # 3. Compute content hash from retrieved text (strongest identity)
        new_content_hash = _sha256(content.content) if content.content else None
        s3_uri = self._s3_uri(file_key)

        # 4. Idempotency gate
        existing_record = self._try_load_memory(s3_uri)
        need_analysis = _content_changed(
            existing_record, new_content_hash, metadata.etag, metadata.size_bytes
        )
        current_model_id = (
            self.embedding_provider.model_id if self.embedding_provider else "none"
        )
        need_embedding = need_analysis or _embedding_stale(existing_record, current_model_id)

        # 5. Relationships (deterministic first pass; semantic pass happens after embedding)
        candidates = [item for item in items if item.key != file_key]
        metadata_dict = metadata.model_dump(mode="json")

        if skip_inference:
            partial_importance = self.scorer.score(metadata, None, [])
            relationships = self.relationship_analyzer.analyze(
                metadata, content.content, candidates
            )
            relationships_list = [r.model_dump(mode="json") for r in relationships]
            return {
                "status": "READY_FOR_INFERENCE",
                "bucket": self.bucket_name,
                "key": file_key,
                "metadata": metadata_dict,
                "relationships": relationships_list,
                "content_preview": (
                    content.content[:100] + "..."
                    if len(content.content) > 100
                    else content.content
                ),
                "content_size": content.size_bytes,
                "target_model": self.agent.model_id,
                "partial_importance_score": partial_importance.model_dump(mode="json"),
                "note": "Stopped before inference as requested.",
            }

        # 6. Semantic Analysis (with idempotency)
        enrichment = None
        if need_analysis:
            analysis = self.agent.analyze_file(
                file_key=file_key,
                content=content.content,
                metadata=metadata_dict,
            )
            
            # Comprehend Enrichment
            enrichment = self.comprehend.analyze_text(content.content)
            
            has_pii = bool(enrichment.get("pii_entities"))
            has_person = any(ent.get("type") == "PERSON" for ent in enrichment.get("entities", []))
            
            if has_pii:
                # Bump sensitivity if explicit PII detected
                if analysis.sensitivity in ["public", "internal"]:
                    analysis.sensitivity = "confidential"
                analysis.reasoning += " [Comprehend Enrichment: Explicit PII detected, increased sensitivity.]"
            elif has_person:
                # Add evidence without escalating sensitivity artificially
                analysis.reasoning += " [Comprehend Enrichment: PERSON entities detected.]"
        else:
            # Reuse cached analysis values — avoid expensive LLM call.
            # recommended_action is read from the persisted record so the
            # cached analysis faithfully represents the original (FIX #6).
            log.debug("Reusing cached semantic analysis for %s", s3_uri)
            analysis = SemanticAnalysisResult(
                key=file_key,
                category=existing_record.category,
                sensitivity=existing_record.sensitivity,
                importance_score=existing_record.importance_score,
                confidence=1.0,
                reasoning="Reused from semantic memory cache (content unchanged).",
                recommended_action=existing_record.recommended_action,
            )

        # 7. Embedding (with model-version idempotency)
        vector: Optional[list] = None
        vector_id = _make_vector_id(self.bucket_name, file_key)

        if need_embedding:
            vector = self._try_embed(content.content)
        else:
            # HIGH-2: Content + model are both unchanged. Retrieve the stored
            # vector so semantic relationship search can still run.
            log.debug("Reusing existing embedding for %s", s3_uri)
            vector = self._try_get_cached_vector(vector_id)

        # 8. Relationship analysis — semantic pass integrated if embedding available
        relationships = self.relationship_analyzer.analyze(
            metadata,
            content.content,
            candidates,
            query_vector=vector,
        )
        relationships_list = [r.model_dump(mode="json") for r in relationships]

        # 9. Importance + Policy
        importance = self.scorer.score(metadata, analysis, relationships)
        decision = self.policy.evaluate(analysis, metadata)

        # 10. Persist memory (HIGH-3 ordering: vector first, then metadata)
        persisted = False
        if need_analysis or need_embedding:
            record = SemanticMemoryRecord(
                s3_uri=s3_uri,
                bucket=self.bucket_name,
                key=file_key,
                content_hash=new_content_hash,
                etag=metadata.etag or "",
                size_bytes=metadata.size_bytes,
                category=analysis.category,
                sensitivity=analysis.sensitivity,
                importance_score=importance.score,
                analysis_timestamp=datetime.now(timezone.utc),
                embedding_model=current_model_id,
                vector_id=vector_id,
                recommended_action=analysis.recommended_action,
            )
            embedding_obj: Optional[Embedding] = None
            if vector is not None:
                embedding_obj = Embedding(
                    vector_id=vector_id,
                    vector=vector,
                    metadata={
                        "s3_uri": s3_uri,
                        "category": analysis.category,
                        "sensitivity": analysis.sensitivity,
                    },
                )
            persisted = self._try_persist(record, embedding_obj)

        # 11. Action request
        requested_action = decision.action
        if requested_action == "TRASH":
            requested_action = "QUARANTINE"

        action_req = ActionRequest(
            s3_uri=s3_uri,
            bucket=self.bucket_name,
            key=file_key,
            requested_action=requested_action,
            reason=" | ".join(decision.reasons),
            risk=decision.risk,
            human_approved=human_approved,
        )

        result_dict = {
            "status": "ANALYZED",
            "bucket": self.bucket_name,
            "key": file_key,
            "metadata": metadata_dict,
            "relationships": relationships_list,
            "analysis": analysis.model_dump(mode="json"),
            "enrichment": enrichment,
            "importance": importance.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "action_request": action_req.model_dump(mode="json"),
            "memory": {
                "reused_analysis": not need_analysis,
                "reused_embedding": not need_embedding,
                "embedding_model": current_model_id,
                "vector_id": vector_id,
                "persisted": persisted,
            },
        }

        if execute_action:
            action_result = self.action_engine.execute(action_req)
            result_dict["action_result"] = action_result.model_dump(mode="json")
            result_dict["status"] = "COMPLETED"

        return result_dict
