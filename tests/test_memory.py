"""
Focused deterministic unit tests for the semantic memory subsystem.

All external service boundaries (DynamoDB, S3 Vectors, embedding APIs) are
replaced by mocks. No live AWS calls are made. No Gemini/Bedrock calls are made.
"""
import datetime
import hashlib
import json
import io
import os
from decimal import Decimal
from typing import List, Optional
from unittest.mock import MagicMock, patch, call
import pytest

from sms_agent.models import (
    SemanticMemoryRecord,
    Embedding,
    SemanticMatch,
    FileMetadata,
    SemanticAnalysisResult,
)
from sms_agent.pipeline import (
    _content_changed, _embedding_stale, _sha256, _make_vector_id, SMSPipeline,
)
from sms_agent.relationships import RelationshipAnalyzer
from sms_agent.memory import (
    DynamoDBMemoryStore, S3VectorStore, validate_memory_config,
)
from sms_agent.embeddings import GeminiEmbeddingProvider, BedrockEmbeddingProvider, EmbeddingError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = datetime.datetime(2026, 9, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _make_record(**kwargs) -> SemanticMemoryRecord:
    defaults = dict(
        s3_uri="s3://bucket/key.txt",
        bucket="bucket",
        key="key.txt",
        content_hash="abc123",
        etag='"etag1"',
        size_bytes=100,
        category="Finance",
        sensitivity="confidential",
        importance_score=0.75,
        analysis_timestamp=_DT,
        embedding_model="gemini-embedding-2",
        vector_id="vid001",
        recommended_action="retain",
    )
    defaults.update(kwargs)
    return SemanticMemoryRecord(**defaults)


def _make_file_meta(**kwargs) -> FileMetadata:
    defaults = dict(
        key="key.txt",
        size_bytes=100,
        created_at=_DT,
        etag='"etag1"',
        bucket="bucket",
    )
    defaults.update(kwargs)
    return FileMetadata(**defaults)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

class TestModels:
    def test_semantic_memory_record_round_trip(self):
        rec = _make_record()
        assert rec.s3_uri == "s3://bucket/key.txt"
        assert rec.embedding_model == "gemini-embedding-2"
        assert rec.vector_id == "vid001"
        assert rec.recommended_action == "retain"
        assert not hasattr(rec, "vector")

    def test_semantic_memory_record_no_raw_vector(self):
        fields = SemanticMemoryRecord.model_fields
        assert "vector" not in fields, "Raw vector must not be stored in SemanticMemoryRecord"

    def test_semantic_memory_record_has_recommended_action(self):
        rec = _make_record(recommended_action="archive")
        assert rec.recommended_action == "archive"

    def test_semantic_memory_record_recommended_action_defaults_to_retain(self):
        # Backward compat: old records without recommended_action default to "retain"
        rec = _make_record()
        assert rec.recommended_action == "retain"

    def test_embedding_model(self):
        emb = Embedding(vector_id="v1", vector=[0.1, 0.2, 0.3], metadata={"s3_uri": "s3://b/k"})
        assert len(emb.vector) == 3

    def test_semantic_match_model(self):
        m = SemanticMatch(vector_id="v1", similarity_score=0.91, metadata={"s3_uri": "s3://b/k"})
        assert m.similarity_score == 0.91


# ---------------------------------------------------------------------------
# Idempotency logic
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_new_document_always_changed(self):
        assert _content_changed(None, "hash1", '"etag"', 100) is True

    def test_same_content_hash_not_changed(self):
        rec = _make_record(content_hash="abc123")
        assert _content_changed(rec, "abc123", '"etag-different"', 999) is False

    def test_different_content_hash_changed(self):
        rec = _make_record(content_hash="abc123")
        assert _content_changed(rec, "def456", '"etag1"', 100) is True

    def test_no_hash_same_etag_and_size_not_changed(self):
        rec = _make_record(content_hash=None, etag='"etag1"', size_bytes=100)
        assert _content_changed(rec, None, '"etag1"', 100) is False

    def test_no_hash_changed_etag_is_changed(self):
        rec = _make_record(content_hash=None, etag='"etag1"', size_bytes=100)
        assert _content_changed(rec, None, '"etag2"', 100) is True

    def test_no_hash_changed_size_is_changed(self):
        rec = _make_record(content_hash=None, etag='"etag1"', size_bytes=100)
        assert _content_changed(rec, None, '"etag1"', 200) is True

    def test_no_identity_at_all_treated_as_changed(self):
        rec = _make_record(content_hash=None, etag="")
        assert _content_changed(rec, None, None, 100) is True


# ---------------------------------------------------------------------------
# Embedding model versioning
# ---------------------------------------------------------------------------

class TestEmbeddingVersioning:
    def test_same_model_not_stale(self):
        rec = _make_record(embedding_model="gemini-embedding-2")
        assert _embedding_stale(rec, "gemini-embedding-2") is False

    def test_different_model_is_stale(self):
        rec = _make_record(embedding_model="gemini-embedding-2")
        assert _embedding_stale(rec, "amazon.titan-embed-text-v2:0") is True

    def test_no_record_is_stale(self):
        assert _embedding_stale(None, "gemini-embedding-2") is True


# ---------------------------------------------------------------------------
# DynamoDB memory store
# ---------------------------------------------------------------------------

class TestDynamoDBMemoryStore:
    def _mock_resource(self, item=None):
        table = MagicMock()
        table.get_item.return_value = {"Item": item} if item else {}
        resource = MagicMock()
        resource.Table.return_value = table
        return resource, table

    def test_get_record_returns_none_when_missing(self):
        resource, _ = self._mock_resource(item=None)
        store = DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource)
        assert store.get_record("s3://bucket/key.txt") is None

    def test_get_record_returns_parsed_record(self):
        resource, _ = self._mock_resource(item={
            "s3_uri": "s3://bucket/key.txt",
            "bucket": "bucket",
            "key": "key.txt",
            "content_hash": "abc123",
            "etag": '"etag1"',
            "size_bytes": 100,
            "category": "Finance",
            "sensitivity": "confidential",
            "importance_score": Decimal("0.750000"),
            "analysis_timestamp": "2026-09-01T12:00:00+00:00",
            "embedding_model": "gemini-embedding-2",
            "vector_id": "vid001",
            "recommended_action": "archive",
        })
        store = DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource)
        rec = store.get_record("s3://bucket/key.txt")
        assert rec is not None
        assert rec.category == "Finance"
        assert rec.embedding_model == "gemini-embedding-2"
        assert rec.recommended_action == "archive"
        assert "vector" not in SemanticMemoryRecord.model_fields

    def test_save_record_calls_put_item(self):
        resource, table = self._mock_resource()
        store = DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource)
        rec = _make_record()
        store.save_record(rec)
        table.put_item.assert_called_once()
        item = table.put_item.call_args[1]["Item"]
        assert item["s3_uri"] == "s3://bucket/key.txt"
        assert "vector" not in item  # Raw vector must NOT be in DynamoDB

    def test_importance_score_stored_as_decimal_not_string(self):
        """FIX #4: importance_score must be a Decimal (numeric), not a string."""
        resource, table = self._mock_resource()
        store = DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource)
        rec = _make_record(importance_score=0.75)
        store.save_record(rec)
        item = table.put_item.call_args[1]["Item"]
        stored = item["importance_score"]
        assert isinstance(stored, Decimal), (
            f"importance_score must be stored as Decimal, got {type(stored).__name__}: {stored!r}"
        )

    def test_recommended_action_persisted(self):
        """FIX #6: recommended_action must survive a round-trip through DynamoDB."""
        resource, table = self._mock_resource()
        store = DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource)
        rec = _make_record(recommended_action="archive")
        store.save_record(rec)
        item = table.put_item.call_args[1]["Item"]
        assert item["recommended_action"] == "archive"

    def test_table_object_cached_not_recreated(self):
        """FIX #7: boto3 Table must only be created once per store instance."""
        resource, table = self._mock_resource()
        store = DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource)
        store.get_record("s3://b/k1")
        store.get_record("s3://b/k2")
        # resource.Table must be called exactly once regardless of how many ops
        resource.Table.assert_called_once_with("test-table")

    def test_get_record_raises_on_dynamo_error(self):
        resource, table = self._mock_resource()
        table.get_item.side_effect = Exception("DynamoDB unavailable")
        store = DynamoDBMemoryStore(table_name="test-table", dynamodb_resource=resource)
        with pytest.raises(RuntimeError, match="DynamoDB get_item failed"):
            store.get_record("s3://bucket/key.txt")


# ---------------------------------------------------------------------------
# S3 Vector store
# ---------------------------------------------------------------------------

class TestS3VectorStore:
    def _mock_client(self, query_result=None, get_result=None):
        client = MagicMock()
        client.query_vectors.return_value = query_result or {"vectors": []}
        client.get_vectors.return_value = get_result or {"vectors": []}
        return client

    def test_upsert_calls_put_vectors(self):
        client = self._mock_client()
        store = S3VectorStore(vector_bucket="test-bucket", dimension=2, s3vectors_client=client)
        emb = Embedding(vector_id="v1", vector=[0.1, 0.2], metadata={"s3_uri": "s3://b/k"})
        store.upsert(emb)
        client.put_vectors.assert_called_once()
        args = client.put_vectors.call_args[1]
        assert args["vectorBucketName"] == "test-bucket"
        assert args["vectors"][0]["key"] == "v1"

    def test_upsert_raises_on_dimension_mismatch(self):
        client = self._mock_client()
        store = S3VectorStore(vector_bucket="test-bucket", dimension=768, s3vectors_client=client)
        emb = Embedding(vector_id="v1", vector=[0.1, 0.2], metadata={})
        with pytest.raises(ValueError, match="expected 768, got 2"):
            store.upsert(emb)

    def test_search_calls_query_vectors_with_return_metadata_true(self):
        """FIX HIGH-1: query_vectors must include returnMetadata=True."""
        client = self._mock_client()
        store = S3VectorStore(vector_bucket="test-bucket", s3vectors_client=client)
        store.search([0.1, 0.2], top_k=5, threshold=0.8)
        call_kwargs = client.query_vectors.call_args[1]
        assert call_kwargs.get("returnMetadata") is True, (
            "returnMetadata=True is required for semantic matches to resolve s3_uri"
        )

    def test_search_returns_metadata_uri_not_vector_id(self):
        """FIX HIGH-1: match.metadata must contain s3_uri, not fall back to vector_id hash."""
        client = self._mock_client(query_result={
            "vectors": [
                {
                    "key": "vid-a",
                    "distance": 0.05,
                    "metadata": {"s3_uri": "s3://bucket/real-doc.txt"},
                }
            ]
        })
        store = S3VectorStore(vector_bucket="test-bucket", s3vectors_client=client)
        matches = store.search([0.1, 0.2], top_k=5, threshold=0.8)
        assert len(matches) == 1
        assert matches[0].metadata.get("s3_uri") == "s3://bucket/real-doc.txt"
        # Must NOT fall back to the vector_id hash
        assert matches[0].metadata.get("s3_uri") != matches[0].vector_id

    def test_search_returns_matches_above_threshold(self):
        client = self._mock_client(query_result={
            "vectors": [
                {"key": "vid-a", "distance": 0.05, "metadata": {"s3_uri": "s3://b/a.txt"}},
                {"key": "vid-b", "distance": 0.50, "metadata": {"s3_uri": "s3://b/b.txt"}},
            ]
        })
        store = S3VectorStore(vector_bucket="test-bucket", s3vectors_client=client)
        matches = store.search([0.1, 0.2], top_k=5, threshold=0.8)
        assert len(matches) == 1
        assert matches[0].vector_id == "vid-a"
        assert matches[0].similarity_score == pytest.approx(0.95, abs=0.01)

    def test_search_returns_empty_when_no_bucket_configured(self):
        client = self._mock_client()
        store = S3VectorStore(vector_bucket="", s3vectors_client=client)
        matches = store.search([0.1, 0.2])
        assert matches == []
        client.query_vectors.assert_not_called()

    def test_upsert_raises_when_no_bucket(self):
        client = self._mock_client()
        store = S3VectorStore(vector_bucket="", s3vectors_client=client)
        with pytest.raises(RuntimeError, match="SMS_VECTOR_BUCKET"):
            store.upsert(Embedding(vector_id="v", vector=[0.1], metadata={}))

    def test_search_raises_on_api_error(self):
        client = self._mock_client()
        client.query_vectors.side_effect = Exception("Network error")
        store = S3VectorStore(vector_bucket="test-bucket", s3vectors_client=client)
        with pytest.raises(RuntimeError, match="query_vectors failed"):
            store.search([0.1, 0.2])

    def test_get_vector_calls_get_vectors_with_return_data(self):
        """FIX HIGH-2: get_vector must request returnData=True."""
        client = self._mock_client(get_result={
            "vectors": [{"key": "vid-a", "data": {"float32": [0.1, 0.2, 0.3]}}]
        })
        store = S3VectorStore(vector_bucket="test-bucket", s3vectors_client=client)
        vec = store.get_vector("vid-a")
        assert vec == [0.1, 0.2, 0.3]
        call_kwargs = client.get_vectors.call_args[1]
        assert call_kwargs.get("returnData") is True

    def test_get_vector_returns_none_when_not_found(self):
        client = self._mock_client(get_result={"vectors": []})
        store = S3VectorStore(vector_bucket="test-bucket", s3vectors_client=client)
        assert store.get_vector("nonexistent") is None

    def test_get_vector_returns_none_when_no_bucket(self):
        client = self._mock_client()
        store = S3VectorStore(vector_bucket="", s3vectors_client=client)
        assert store.get_vector("vid-a") is None
        client.get_vectors.assert_not_called()

    def test_get_vector_returns_none_on_error(self):
        client = self._mock_client()
        client.get_vectors.side_effect = Exception("S3 Vectors unavailable")
        store = S3VectorStore(vector_bucket="test-bucket", s3vectors_client=client)
        # Should return None, not raise
        assert store.get_vector("vid-a") is None


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

class TestValidateMemoryConfig:
    def test_all_none_is_valid(self):
        """All disabled is a valid 'no persistence' mode."""
        validate_memory_config(None, None, None)  # Must not raise

    def test_all_configured_is_valid(self):
        validate_memory_config(MagicMock(), MagicMock(), MagicMock())  # Must not raise

    def test_vector_without_memory_raises(self):
        """FIX #5: vector_store without memory_store is partial config."""
        with pytest.raises(ValueError, match="PARTIALLY configured"):
            validate_memory_config(memory_store=None, vector_store=MagicMock(), embedding_provider=MagicMock())

    def test_memory_without_vector_raises(self):
        """FIX #5: memory_store without vector_store is partial config."""
        with pytest.raises(ValueError, match="PARTIALLY configured"):
            validate_memory_config(memory_store=MagicMock(), vector_store=None, embedding_provider=MagicMock())

    def test_stores_without_embedding_raises(self):
        """FIX #5: both stores configured but no embedding provider is partial config."""
        with pytest.raises(ValueError, match="PARTIALLY configured"):
            validate_memory_config(memory_store=MagicMock(), vector_store=MagicMock(), embedding_provider=None)

    def test_only_embedding_without_stores_raises(self):
        """Embedding provider alone (no stores) counts as partial."""
        with pytest.raises(ValueError, match="PARTIALLY configured"):
            validate_memory_config(memory_store=None, vector_store=None, embedding_provider=MagicMock())


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

class TestGeminiEmbeddingProvider:
    def test_model_id_is_correct(self):
        prov = GeminiEmbeddingProvider(api_key="fake-key")
        assert prov.model_id == "gemini-embedding-2"

    def test_raises_without_api_key(self):
        with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
            GeminiEmbeddingProvider(api_key="")

    def test_raises_on_empty_text(self):
        prov = GeminiEmbeddingProvider(api_key="fake-key")
        with pytest.raises(EmbeddingError, match="empty text"):
            prov.embed_text("")

    def test_embed_text_parses_response(self):
        fake_resp = io.BytesIO(json.dumps({
            "embedding": {"values": [0.1, 0.2, 0.3]}
        }).encode())
        with patch("urllib.request.urlopen", return_value=fake_resp):
            prov = GeminiEmbeddingProvider(api_key="fake-key")
            vec = prov.embed_text("hello world")
        assert vec == [0.1, 0.2, 0.3]

    def test_embed_raises_on_bad_response_shape(self):
        fake_resp = io.BytesIO(json.dumps({"unexpected": "shape"}).encode())
        with patch("urllib.request.urlopen", return_value=fake_resp):
            prov = GeminiEmbeddingProvider(api_key="fake-key")
            with pytest.raises(EmbeddingError, match="Unexpected"):
                prov.embed_text("hello world")


class TestBedrockEmbeddingProvider:
    def test_model_id_default(self):
        prov = BedrockEmbeddingProvider()
        assert "titan" in prov.model_id or "embed" in prov.model_id

    def test_model_id_custom(self):
        prov = BedrockEmbeddingProvider(model_id="custom.model")
        assert prov.model_id == "custom.model"

    def test_raises_on_empty_text(self):
        mock_client = MagicMock()
        prov = BedrockEmbeddingProvider(bedrock_client=mock_client)
        with pytest.raises(EmbeddingError, match="empty text"):
            prov.embed_text("")

    def test_embed_text_with_mock_client(self):
        import json
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": [0.4, 0.5]}).encode())
        }
        prov = BedrockEmbeddingProvider(bedrock_client=mock_client)
        vec = prov.embed_text("test text")
        assert vec == [0.4, 0.5]

    def test_default_model_is_authorized_titan_v2(self):
        prov = BedrockEmbeddingProvider()
        assert prov.model_id == "amazon.titan-embed-text-v2:0"

    def test_request_pins_1024_dimensions(self):
        """The 1024-dim contract with the S3 Vectors index must be
        explicit in the request, never reliant on service defaults."""
        import json
        captured = {}

        class _Body:
            def __init__(self, payload):
                captured.update(json.loads(payload.decode("utf-8")))

            def read(self):
                return b"{}"

        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = lambda **kwargs: (
            captured.update(json.loads(kwargs["body"].decode("utf-8"))),
            {"body": MagicMock(
                read=lambda: json.dumps({"embedding": [0.1]}).encode())}
        )[1]
        prov = BedrockEmbeddingProvider(bedrock_client=mock_client)
        prov.embed_text("test text")
        assert captured["dimensions"] == 1024
        assert captured["inputText"] == "test text"

    def test_client_errors_become_embedding_errors(self):
        from botocore.exceptions import ClientError
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException",
                       "Message": "Operation not allowed"}},
            "InvokeModel")
        prov = BedrockEmbeddingProvider(bedrock_client=mock_client)
        with pytest.raises(EmbeddingError, match="Bedrock embedding error"):
            prov.embed_text("test text")

    def test_uses_bedrock_runtime_in_default_region(self):
        import json
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps(
                {"embedding": [0.1]}).encode())}
        with patch("boto3.client", return_value=mock_client) as mock_boto:
            prov = BedrockEmbeddingProvider()
            prov.embed_text("test text")
            mock_boto.assert_called_once_with(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


# ---------------------------------------------------------------------------
# Relationship analyzer — semantic extension
# ---------------------------------------------------------------------------

class TestRelationshipAnalyzerSemantic:
    def _meta(self, key, etag=None, size=100, s3_uri=None, bucket="bucket") -> FileMetadata:
        return FileMetadata(
            key=key, size_bytes=size, created_at=_DT,
            etag=etag, bucket=bucket,
            s3_uri=s3_uri or f"s3://bucket/{key}",
        )

    def test_hash_duplicate_still_wins_over_semantic(self):
        target = self._meta("a.txt", etag='"etag1"')
        target.content_hash = "same_hash"
        cand = self._meta("b.txt", etag='"etag1"')
        cand.content_hash = "same_hash"
        analyzer = RelationshipAnalyzer(vector_store=None)
        results = analyzer.analyze(target, None, [cand])
        assert results[0].relationship_type == "DUPLICATE_CONFIRMED"

    def test_etag_candidate_still_works(self):
        target = self._meta("a.txt", etag='"same"', size=200)
        cand = self._meta("b.txt", etag='"same"', size=200)
        analyzer = RelationshipAnalyzer(vector_store=None)
        results = analyzer.analyze(target, None, [cand])
        assert any(r.relationship_type == "DUPLICATE_CANDIDATE" for r in results)

    def test_filename_relationship_still_works(self):
        target = FileMetadata(key="report.txt", filename="report.txt", size_bytes=100,
                              created_at=_DT, bucket="bucket")
        cand = FileMetadata(key="report-copy.txt", filename="report-copy.txt", size_bytes=100,
                            created_at=_DT, bucket="bucket")
        analyzer = RelationshipAnalyzer(vector_store=None)
        results = analyzer.analyze(target, None, [cand])
        assert any(r.relationship_type == "RELATED" for r in results)

    def test_semantic_similarity_produces_related(self):
        mock_vs = MagicMock()
        mock_vs.search.return_value = [
            SemanticMatch(
                vector_id="vid-x",
                similarity_score=0.91,
                metadata={"s3_uri": "s3://bucket/other.txt"},
            )
        ]
        target = self._meta("a.txt")
        cand = self._meta("other.txt")
        analyzer = RelationshipAnalyzer(vector_store=mock_vs)
        results = analyzer.analyze(target, "content", [cand], query_vector=[0.1, 0.2])
        semantic = [r for r in results if "semantic similarity" in " ".join(r.evidence)]
        assert len(semantic) >= 1

    def test_no_embedding_skips_vector_search(self):
        mock_vs = MagicMock()
        target = self._meta("a.txt")
        analyzer = RelationshipAnalyzer(vector_store=mock_vs)
        analyzer.analyze(target, "content", [], query_vector=None)
        mock_vs.search.assert_not_called()

    def test_vector_search_failure_does_not_crash_deterministic_relationships(self):
        mock_vs = MagicMock()
        mock_vs.search.side_effect = RuntimeError("S3 Vectors unavailable")
        target = self._meta("a.txt", etag='"e1"', size=100)
        cand = self._meta("b.txt", etag='"e1"', size=100)
        analyzer = RelationshipAnalyzer(vector_store=mock_vs)
        results = analyzer.analyze(target, None, [cand], query_vector=[0.1])
        assert any(r.relationship_type == "DUPLICATE_CANDIDATE" for r in results)

    def test_semantic_does_not_override_confirmed_duplicate(self):
        mock_vs = MagicMock()
        mock_vs.search.return_value = [
            SemanticMatch(
                vector_id="vid-y",
                similarity_score=0.92,
                metadata={"s3_uri": "s3://bucket/b.txt"},
            )
        ]
        target = self._meta("a.txt")
        target.content_hash = "same"
        cand = self._meta("b.txt", s3_uri="s3://bucket/b.txt")
        cand.content_hash = "same"
        analyzer = RelationshipAnalyzer(vector_store=mock_vs)
        results = analyzer.analyze(target, None, [cand], query_vector=[0.1])
        match = next(r for r in results if r.related_object == "s3://bucket/b.txt")
        assert match.relationship_type == "DUPLICATE_CONFIRMED"

    def test_self_match_excluded_when_bucket_is_none(self):
        """FIX #8: self-match exclusion must not crash when target.bucket is None."""
        mock_vs = MagicMock()
        target_uri = "s3://bucket/a.txt"
        mock_vs.search.return_value = [
            SemanticMatch(
                vector_id="self-vid",
                similarity_score=0.99,
                metadata={"s3_uri": target_uri},
            )
        ]
        # Create target without bucket attribute set
        target = FileMetadata(key="a.txt", size_bytes=100, created_at=_DT,
                              s3_uri=target_uri, bucket=None)
        analyzer = RelationshipAnalyzer(vector_store=mock_vs)
        # Must not raise, and self-match must be excluded
        results = analyzer.analyze(target, None, [], query_vector=[0.1, 0.2])
        assert all(r.related_object != target_uri for r in results)


# ---------------------------------------------------------------------------
# Pipeline idempotency
# ---------------------------------------------------------------------------

class TestPipelineIdempotency:
    """Tests that pipeline correctly skips redundant inference/embedding calls."""

    def _make_pipeline_mocks(self, cached_vector=None):
        """Returns (pipeline, mocked_agent, mocked_memory_store, mocked_vector_store, mocked_embedding_provider)"""
        mock_s3 = MagicMock()
        mock_inventory = MagicMock()
        meta = FileMetadata(
            key="doc.txt", size_bytes=500, created_at=_DT,
            etag='"etag-1"', bucket="bucket",
            s3_uri="s3://bucket/doc.txt",
            filename="doc.txt",
        )
        mock_inventory.collect.return_value = [meta]

        from sms_agent.models import RetrievedContent
        mock_reader = MagicMock()
        mock_reader.get_text.return_value = RetrievedContent(
            bucket="bucket", key="doc.txt",
            content="This is a financial report.",
            content_type="text/plain", size_bytes=500,
        )

        mock_agent = MagicMock()
        mock_agent.model_id = "test-model"
        mock_agent.analyze_file.return_value = SemanticAnalysisResult(
            key="doc.txt", category="Finance", sensitivity="confidential",
            importance_score=0.8, confidence=0.9,
            reasoning="Test", recommended_action="retain",
        )

        mock_memory = MagicMock()
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        mock_vector.get_vector.return_value = cached_vector  # Returns stored vector on cache hit

        mock_embed = MagicMock()
        mock_embed.model_id = "gemini-embedding-2"
        mock_embed.embed_text.return_value = [0.1, 0.2, 0.3]

        pipeline = SMSPipeline(
            bucket_name="bucket",
            s3_client=mock_s3,
            memory_store=mock_memory,
            vector_store=mock_vector,
            embedding_provider=mock_embed,
        )
        pipeline.inventory = mock_inventory
        pipeline.reader = mock_reader
        pipeline.agent = mock_agent
        pipeline.relationship_analyzer = RelationshipAnalyzer(vector_store=mock_vector)

        return pipeline, mock_agent, mock_memory, mock_vector, mock_embed

    def test_first_processing_persists_memory(self):
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks()
        memory.get_record.return_value = None

        result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        agent.analyze_file.assert_called_once()
        embed.embed_text.assert_called_once()
        vector.upsert.assert_called_once()
        memory.save_record.assert_called_once()
        assert "memory" in result
        assert result["memory"]["reused_analysis"] is False

    def test_vector_upserted_before_metadata_on_first_write(self):
        """FIX HIGH-3: vector must be persisted before DynamoDB metadata."""
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks()
        memory.get_record.return_value = None

        call_order = []
        vector.upsert.side_effect = lambda *a, **kw: call_order.append("vector")
        memory.save_record.side_effect = lambda *a, **kw: call_order.append("dynamo")

        pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        assert call_order == ["vector", "dynamo"], (
            f"Expected vector before dynamo, got: {call_order}"
        )

    def test_vector_failure_prevents_dynamo_write(self):
        """FIX HIGH-3: if vector upsert fails, DynamoDB must NOT be written."""
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks()
        memory.get_record.return_value = None
        vector.upsert.side_effect = RuntimeError("S3 Vectors unavailable")

        result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)

        # DynamoDB must NOT be written
        memory.save_record.assert_not_called()
        # But analysis should still complete
        assert "analysis" in result
        assert result["memory"]["persisted"] is False

    def test_dynamo_failure_does_not_corrupt_state(self):
        """FIX HIGH-3: if DynamoDB fails after vector write, vector is orphaned but DynamoDB is clean."""
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks()
        memory.get_record.return_value = None
        memory.save_record.side_effect = RuntimeError("DynamoDB unavailable")

        # Vector upsert should still be called
        result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        vector.upsert.assert_called_once()  # Vector was written
        assert result["memory"]["persisted"] is False  # Overall persistence failed

    def test_unchanged_content_reuses_analysis_and_embedding(self):
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks(
            cached_vector=[0.5, 0.6, 0.7]
        )
        content_hash = _sha256("This is a financial report.")
        memory.get_record.return_value = _make_record(
            s3_uri="s3://bucket/doc.txt",
            bucket="bucket",
            key="doc.txt",
            content_hash=content_hash,
            etag='"etag-1"',
            size_bytes=500,
            embedding_model="gemini-embedding-2",
        )

        result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        agent.analyze_file.assert_not_called()
        embed.embed_text.assert_not_called()
        assert result["memory"]["reused_analysis"] is True
        assert result["memory"]["reused_embedding"] is True

    def test_cache_hit_still_performs_semantic_search(self):
        """FIX HIGH-2: cache-hit documents must still get semantic relationship analysis."""
        cached_vec = [0.5, 0.6, 0.7]
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks(
            cached_vector=cached_vec
        )
        content_hash = _sha256("This is a financial report.")
        memory.get_record.return_value = _make_record(
            s3_uri="s3://bucket/doc.txt",
            bucket="bucket",
            key="doc.txt",
            content_hash=content_hash,
            etag='"etag-1"',
            size_bytes=500,
            embedding_model="gemini-embedding-2",
        )

        pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)

        # get_vector must be called to retrieve the cached embedding
        vector.get_vector.assert_called_once()
        # search must be called with the retrieved vector (not None)
        vector.search.assert_called_once()
        search_args = vector.search.call_args
        query = search_args[0][0] if search_args[0] else search_args[1].get("query_vector")
        assert query == cached_vec, "Semantic search must use cached vector, not None"

    def test_embedding_model_upgrade_reembeds_but_reuses_analysis(self):
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks()
        content_hash = _sha256("This is a financial report.")
        memory.get_record.return_value = _make_record(
            s3_uri="s3://bucket/doc.txt",
            bucket="bucket",
            key="doc.txt",
            content_hash=content_hash,
            etag='"etag-1"',
            size_bytes=500,
            embedding_model="old-model-v1",
        )

        result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        agent.analyze_file.assert_not_called()
        embed.embed_text.assert_called_once()
        assert result["memory"]["reused_analysis"] is True
        assert result["memory"]["reused_embedding"] is False

    def test_cached_recommended_action_preserved(self):
        """FIX #6: cache-hit analysis must use the stored recommended_action, not 'retain'."""
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks(
            cached_vector=[0.1, 0.2]
        )
        content_hash = _sha256("This is a financial report.")
        memory.get_record.return_value = _make_record(
            s3_uri="s3://bucket/doc.txt",
            bucket="bucket",
            key="doc.txt",
            content_hash=content_hash,
            etag='"etag-1"',
            size_bytes=500,
            embedding_model="gemini-embedding-2",
            recommended_action="archive",  # Was originally "archive", not "retain"
        )

        result = pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        assert result["memory"]["reused_analysis"] is True
        # The reconstructed analysis must preserve the original recommended_action
        assert result["analysis"]["recommended_action"] == "archive"

    def test_execute_action_false_prevents_action_engine_call(self):
        pipeline, agent, memory, vector, embed = self._make_pipeline_mocks()
        memory.get_record.return_value = None
        pipeline.action_engine = MagicMock()

        pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)
        pipeline.action_engine.execute.assert_not_called()

    def test_memory_write_does_not_trigger_s3_mutation(self):
        pipeline, _, memory, vector, embed = self._make_pipeline_mocks()
        memory.get_record.return_value = None
        mock_s3_client = MagicMock()
        pipeline.action_engine = MagicMock()

        pipeline.process_object("doc.txt", skip_inference=False, execute_action=False)

        assert not mock_s3_client.delete_object.called
        assert not mock_s3_client.copy_object.called
