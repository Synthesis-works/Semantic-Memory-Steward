"""
Persistent semantic memory for Semantic Memory Steward.

DynamoDBMemoryStore  â€” metadata records in DynamoDB.
S3VectorStore        â€” embeddings via Amazon S3 Vectors API.

Neither implementation calls AWS during __init__; clients are injected for
testability. Production code obtains real clients lazily.

Persistence ordering contract (HIGH-3):
  1. Upsert vector to S3 Vectors first.
  2. Only if that succeeds, write metadata to DynamoDB.
  This ensures DynamoDB never claims a vector exists that was not persisted.
  Orphaned vectors (vector written but DynamoDB write fails) are possible but
  acceptable for an MVP â€” they are inert and do not cause incorrect results.
"""
import os
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List

from .models import SemanticMemoryRecord, Embedding, SemanticMatch

log = logging.getLogger(__name__)

_DYNAMO_TABLE_ENV = "SMS_DYNAMO_TABLE"
_DEFAULT_DYNAMO_TABLE = "sms-semantic-memory"

_S3V_BUCKET_ENV = "SMS_VECTOR_BUCKET"
_S3V_INDEX_ENV = "SMS_VECTOR_INDEX"
_DEFAULT_S3V_INDEX = "sms-embeddings"


def validate_memory_config(
    memory_store=None,
    vector_store=None,
    embedding_provider=None,
) -> None:
    """
    Validate that the semantic-memory configuration is either fully enabled or
    fully disabled. Partial configuration is logged as a clear warning.

    Raises:
        ValueError â€” if the configuration is partially enabled in a way that
                     would produce inconsistent state (e.g. vector store
                     configured without a memory store).

    Call this at application startup or before the first pipeline run.
    Does NOT create any AWS resources.
    """
    has_memory = memory_store is not None
    has_vector = vector_store is not None
    has_embed = embedding_provider is not None

    if not has_memory and not has_vector and not has_embed:
        log.info(
            "Semantic memory layer is DISABLED "
            "(no memory_store, vector_store, or embedding_provider configured). "
            "Analysis will proceed without persistence."
        )
        return

    issues = []
    if has_vector and not has_memory:
        issues.append(
            "vector_store is configured but memory_store is missing â€” "
            "vectors will be orphaned with no associated metadata."
        )
    if has_memory and not has_vector:
        issues.append(
            "memory_store is configured but vector_store is missing â€” "
            "DynamoDB records will carry stale/invalid vector_id references."
        )
    if (has_memory or has_vector) and not has_embed:
        issues.append(
            "memory_store or vector_store is configured but embedding_provider is missing â€” "
            "new documents cannot be embedded."
        )
    if has_embed and not has_memory and not has_vector:
        issues.append(
            "embedding_provider is configured but memory_store and vector_store are missing â€” "
            "embeddings will be generated but nowhere to store them."
        )

    if issues:
        msg = "Semantic memory is PARTIALLY configured:\n" + "\n".join(
            f"  - {i}" for i in issues
        )
        log.warning(msg)
        raise ValueError(msg)

    log.info(
        "Semantic memory layer: DynamoDB=%s, VectorStore=%s, EmbeddingProvider=%s",
        type(memory_store).__name__,
        type(vector_store).__name__,
        type(embedding_provider).__name__,
    )


# ---------------------------------------------------------------------------
# DynamoDB memory store
# ---------------------------------------------------------------------------

class DynamoDBMemoryStore:
    """
    Stores/retrieves SemanticMemoryRecord objects in DynamoDB.

    The raw vector is NOT stored here â€” only a vector_id reference.
    Table must be provisioned externally before use:
        Table name   : SMS_DYNAMO_TABLE env var (default: sms-semantic-memory)
        Partition key: s3_uri (String)
        Billing      : PAY_PER_REQUEST

    boto3 reuse: The DynamoDB resource and Table object are created once per
    store instance on first use, not recreated on every call.
    """

    def __init__(self, table_name: Optional[str] = None, dynamodb_resource=None):
        self._table_name = table_name or os.environ.get(
            _DYNAMO_TABLE_ENV, _DEFAULT_DYNAMO_TABLE
        )
        self._resource = dynamodb_resource  # Injected for testing
        self._table_obj = None             # Lazy, cached Table reference

    def _table(self):
        """Return a cached DynamoDB Table object (creates once per instance)."""
        if self._table_obj is not None:
            return self._table_obj
        resource = self._resource
        if resource is None:
            import boto3
            resource = boto3.resource(
                "dynamodb",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
        self._table_obj = resource.Table(self._table_name)
        return self._table_obj

    def get_record(self, s3_uri: str) -> Optional[SemanticMemoryRecord]:
        """
        Fetch an existing record by s3_uri.
        Returns None if not found. Raises on unexpected errors.
        """
        try:
            resp = self._table().get_item(Key={"s3_uri": s3_uri})
        except Exception as exc:
            raise RuntimeError(f"DynamoDB get_item failed for {s3_uri!r}: {exc}") from exc

        item = resp.get("Item")
        if item is None:
            return None

        try:
            return SemanticMemoryRecord(
                s3_uri=item["s3_uri"],
                bucket=item["bucket"],
                key=item["key"],
                content_hash=item.get("content_hash") or None,
                etag=item["etag"],
                size_bytes=int(item["size_bytes"]),
                category=item["category"],
                sensitivity=item["sensitivity"],
                importance_score=float(item["importance_score"]),  # Decimal â†’ float
                analysis_timestamp=datetime.fromisoformat(item["analysis_timestamp"]),
                embedding_model=item["embedding_model"],
                vector_id=item["vector_id"],
                recommended_action=item.get("recommended_action", "retain"),
                human_decision=item.get("human_decision"),
            )
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Malformed DynamoDB item for {s3_uri!r}: {exc}"
            ) from exc

    def save_record(self, record: SemanticMemoryRecord) -> None:
        """
        Persist a SemanticMemoryRecord.
        Uses put_item (upsert semantics â€” idempotent for the same s3_uri).

        importance_score is stored as a DynamoDB Number (Decimal) so numeric
        comparators remain available for future queries.
        """
        importance_decimal = Decimal(str(record.importance_score)).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
        item = {
            "s3_uri": record.s3_uri,
            "bucket": record.bucket,
            "key": record.key,
            "etag": record.etag,
            "size_bytes": record.size_bytes,
            "category": record.category,
            "sensitivity": record.sensitivity,
            "importance_score": importance_decimal,  # Numeric, not string
            "analysis_timestamp": record.analysis_timestamp.isoformat(),
            "embedding_model": record.embedding_model,
            "vector_id": record.vector_id,
            "recommended_action": record.recommended_action,
        }
        if record.content_hash:
            item["content_hash"] = record.content_hash
        if record.human_decision:
            item["human_decision"] = record.human_decision

        try:
            self._table().put_item(Item=item)
        except Exception as exc:
            raise RuntimeError(
                f"DynamoDB put_item failed for {record.s3_uri!r}: {exc}"
            ) from exc

    def record_human_decision(self, s3_uri: str, decision: str) -> SemanticMemoryRecord:
        """
        Persist a human review outcome on an existing record.

        Narrow UI-integration seam: the dashboard calls this after a human
        approves KEEP so the decision survives reruns and rescans. Policy
        output (recommended_action) is never modified here.
        """
        if decision != "KEEP":
            raise ValueError(
                f"record_human_decision: unsupported decision {decision!r} "
                "(only 'KEEP' is recorded as a human decision)"
            )
        record = self.get_record(s3_uri)
        if record is None:
            raise ValueError(
                f"record_human_decision: no record for {s3_uri!r}"
            )
        updated = record.model_copy(update={"human_decision": decision})
        self.save_record(updated)
        return updated


# ---------------------------------------------------------------------------
# Amazon S3 Vectors store
# ---------------------------------------------------------------------------

class S3VectorStore:
    """
    Stores and queries embeddings via Amazon S3 Vectors.

    Requires:
        SMS_VECTOR_BUCKET â€” name of the S3 Vectors bucket (must exist)
        SMS_VECTOR_INDEX  â€” index name inside that bucket (default: sms-embeddings)

    The bucket/index must be provisioned externally before use.
    This class is purely a data-access layer; it performs no AWS provisioning.

    IMPORTANT â€” distance metric assumption:
        S3 Vectors supports 'cosine' and 'euclidean' distance metrics,
        configured at index-creation time. This implementation assumes 'cosine'
        distance (lower = more similar), where similarity = 1 - distance.
        The index MUST be created with distanceMetric='cosine' for threshold
        comparisons to be meaningful.

    boto3 S3 Vectors API used:
        put_vectors   â€” upsert one or more vectors
        query_vectors â€” approximate nearest-neighbour search (returnMetadata=True)
        get_vectors   â€” retrieve stored vector data by key (for cache-hit reuse)
    """

    def __init__(
        self,
        vector_bucket: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        s3vectors_client=None,
    ):
        self._bucket = vector_bucket or os.environ.get(_S3V_BUCKET_ENV, "")
        self._index = index_name or os.environ.get(_S3V_INDEX_ENV, _DEFAULT_S3V_INDEX)

        # Determine dimension explicitly
        env_dim = os.environ.get("SMS_VECTOR_DIMENSION")
        if dimension is not None:
            self._dimension = dimension
        elif env_dim:
            self._dimension = int(env_dim)
        else:
            self._dimension = 768

        self._client = s3vectors_client  # Injected for testing

    def _s3v(self):
        if self._client is not None:
            return self._client
        import boto3
        return boto3.client(
            "s3vectors",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def upsert(self, embedding: Embedding) -> None:
        """
        Store or replace a vector embedding.
        vector_id is used as the stable key.
        metadata is stored alongside the vector for filtering/retrieval.
        """
        if not self._bucket:
            raise RuntimeError(
                f"S3VectorStore requires a vector bucket name. Set {_S3V_BUCKET_ENV}."
            )

        actual_dim = len(embedding.vector)
        if actual_dim != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {actual_dim}. "
                f"Reconfigure SMS_VECTOR_DIMENSION or use a compatible model."
            )

        vector_item = {
            "key": embedding.vector_id,
            "data": {"float32": embedding.vector},
        }
        if embedding.metadata:
            vector_item["metadata"] = embedding.metadata

        try:
            self._s3v().put_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                vectors=[vector_item],
            )
        except Exception as exc:
            raise RuntimeError(
                f"S3 Vectors put_vectors failed for vector_id={embedding.vector_id!r}: {exc}"
            ) from exc

    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """
        Retrieve the raw float vector for a known vector_id.
        Used by the pipeline to reuse an existing embedding on cache hits
        without regenerating it via the embedding API.

        Returns None if the vector is not found or an error occurs.
        """
        if not self._bucket:
            return None

        try:
            resp = self._s3v().get_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                keys=[vector_id],
                returnData=True,
            )
        except Exception as exc:
            log.warning("S3 Vectors get_vectors failed for %r: %s", vector_id, exc)
            return None

        vectors = resp.get("vectors", [])
        if not vectors:
            return None

        data = vectors[0].get("data", {})
        return data.get("float32")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.8,
    ) -> List[SemanticMatch]:
        """
        Query for approximate nearest neighbours above the similarity threshold.

        returnMetadata=True is required so metadata (including s3_uri) is
        returned alongside each match. Without it, metadata is always empty
        and matches cannot be resolved back to S3 URIs.

        Returns an empty list if the bucket is not configured.
        """
        if not self._bucket:
            log.warning("SMS_VECTOR_BUCKET not set â€” vector search unavailable")
            return []

        try:
            resp = self._s3v().query_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                queryVector={"float32": query_vector},
                topK=top_k,
                returnDistance=True,
                returnMetadata=True,   # FIX HIGH-1: required to resolve s3_uri from match
            )
        except Exception as exc:
            raise RuntimeError(f"S3 Vectors query_vectors failed: {exc}") from exc

        matches: List[SemanticMatch] = []
        for hit in resp.get("vectors", []):
            # Assumes index was created with distanceMetric='cosine'.
            # cosine distance is in [0, 2]; for normalised vectors it's in [0, 1].
            # similarity = 1 - distance (higher = more similar).
            distance = hit.get("distance", 1.0)
            similarity = max(0.0, 1.0 - distance)
            if similarity >= threshold:
                metadata = hit.get("metadata") or {}
                matches.append(
                    SemanticMatch(
                        vector_id=hit["key"],
                        similarity_score=round(similarity, 4),
                        metadata=metadata,
                    )
                )

        return matches
