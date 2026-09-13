"""Vector infrastructure tests: correct bucket, 1024-dim contract.

The app must hand S3VectorStore the real vector bucket (not the document
bucket), and the store/index dimensionality must agree at 1024 for the
Titan embedding path. All AWS boundaries are mocked.
"""
import os
from unittest.mock import MagicMock, patch

from streamlit import cache_data
from streamlit.testing.v1 import AppTest

from sms_agent.memory import S3VectorStore
from sms_agent.embeddings import BedrockEmbeddingProvider, GeminiEmbeddingProvider
from sms_agent.pipeline import SMSPipeline
from sms_agent.models import FileMetadata, RetrievedContent, SemanticAnalysisResult

APP = r"D:\SMS\app.py"
TIMEOUT = 90
VECTOR_BUCKET = "sms-semantic-vectors-527557823928"


def _init_pipeline(extra_env=None):
    """Run the real app init (no preset pipeline) with boto3 mocked.

    Proves which bucket the application configuration actually hands
    to S3VectorStore. No network is touched.
    """
    cache_data.clear()
    env = dict(extra_env or {})
    with patch.dict(os.environ, env, clear=False):
        for key in ("SMS_VECTOR_BUCKET",):
            if key not in env:
                os.environ.pop(key, None)
        with patch("boto3.client"), patch("boto3.resource"):
            at = AppTest.from_file(APP)
            at.run(timeout=TIMEOUT)
    assert not at.exception, f"init crashed: {at.exception}"
    return at.session_state["pipeline"]


def test_app_configures_real_vector_bucket_by_default():
    pipeline = _init_pipeline()
    assert pipeline.vector_store._bucket == VECTOR_BUCKET


def test_app_vector_bucket_env_override():
    pipeline = _init_pipeline({"SMS_VECTOR_BUCKET": "custom-vectors"})
    assert pipeline.vector_store._bucket == "custom-vectors"


def test_store_accepts_1024_dim_vectors():
    client = MagicMock()
    store = S3VectorStore(vector_bucket="b", index_name="i", dimension=1024,
                          s3vectors_client=client)
    from sms_agent.models import Embedding
    store.upsert(Embedding(vector_id="v", vector=[0.1] * 1024,
                           metadata={}))
    client.put_vectors.assert_called_once()


def test_store_rejects_768_dim_vectors_when_configured_1024():
    client = MagicMock()
    store = S3VectorStore(vector_bucket="b", index_name="i", dimension=1024,
                          s3vectors_client=client)
    from sms_agent.models import Embedding
    import pytest
    with pytest.raises(ValueError, match="expected 1024, got 768"):
        store.upsert(Embedding(vector_id="v", vector=[0.1] * 768,
                               metadata={}))


def test_empty_index_search_returns_empty():
    client = MagicMock()
    client.query_vectors.return_value = {"vectors": []}
    store = S3VectorStore(vector_bucket="b", index_name="i", dimension=1024,
                          s3vectors_client=client)
    assert store.search([0.1] * 1024) == []


def test_app_default_embedding_provider_is_bedrock_titan():
    """Default production wiring: Bedrock Titan V2 for embeddings, 1024-dim
    store dims matching the live index, no Gemini forced anywhere."""
    pipeline = _init_pipeline()
    assert isinstance(pipeline.embedding_provider, BedrockEmbeddingProvider)
    assert pipeline.embedding_provider.model_id == "amazon.titan-embed-text-v2:0"
    assert pipeline.vector_store._dimension == 1024


def test_app_gemini_embedding_provider_explicit_selection():
    pipeline = _init_pipeline({
        "SMS_EMBEDDING_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
    })
    assert isinstance(pipeline.embedding_provider, GeminiEmbeddingProvider)
    assert pipeline.embedding_provider.model_id == "gemini-embedding-2"


def test_app_disabled_embedding_provider_means_no_vector_operations():
    pipeline = _init_pipeline({"SMS_EMBEDDING_PROVIDER": "none"})
    assert pipeline.embedding_provider is None


def _seam_pipeline(dim_vector):
    """Real SMSPipeline + real S3VectorStore(1024) + real BedrockEmbeddingProvider
    with mocked clients so the full embed->upsert seam is exercised end to end."""
    from datetime import datetime, timezone
    import json as _json

    _DT = datetime.now(timezone.utc)
    mock_bedrock = MagicMock()
    mock_bedrock.invoke_model.return_value = {
        "body": MagicMock(read=lambda: _json.dumps(
            {"embedding": dim_vector}).encode())}

    mock_s3vectors = MagicMock()
    mock_s3vectors.query_vectors.return_value = {"vectors": []}
    store = S3VectorStore(vector_bucket="vec-bucket", index_name="idx",
                          dimension=1024, s3vectors_client=mock_s3vectors)
    embed = BedrockEmbeddingProvider(bedrock_client=mock_bedrock)

    meta = FileMetadata(key="doc.txt", size_bytes=500, created_at=_DT,
                        etag='"etag-1"', bucket="bucket",
                        s3_uri="s3://bucket/doc.txt", filename="doc.txt")
    mock_inventory = MagicMock()
    mock_inventory.collect.return_value = [meta]
    mock_reader = MagicMock()
    mock_reader.get_text.return_value = RetrievedContent(
        bucket="bucket", key="doc.txt",
        content="This is a financial report.",
        content_type="text/plain", size_bytes=500)
    mock_agent = MagicMock()
    mock_agent.model_id = "test-model"
    mock_agent.analyze_file.return_value = SemanticAnalysisResult(
        key="doc.txt", category="Finance", sensitivity="confidential",
        importance_score=0.8, confidence=0.9, reasoning="Test",
        recommended_action="retain")
    mock_memory = MagicMock()
    mock_memory.get_record.return_value = None

    pipeline = SMSPipeline(
        bucket_name="bucket",
        s3_client=MagicMock(),
        memory_store=mock_memory,
        vector_store=store,
        embedding_provider=embed,
    )
    pipeline.inventory = mock_inventory
    pipeline.reader = mock_reader
    pipeline.agent = mock_agent
    pipeline.comprehend = MagicMock()
    return pipeline, mock_memory, mock_s3vectors, mock_bedrock


def test_titan_embedding_flows_through_pipeline_to_vector_store():
    dim_vector = [0.01 * (i % 10) + 0.001 * i for i in range(1024)]
    pipeline, memory, s3vectors, bedrock = _seam_pipeline(dim_vector)

    result = pipeline.process_object("doc.txt", skip_inference=False,
                                     execute_action=False)

    assert result["memory"]["embedding_model"] == "amazon.titan-embed-text-v2:0"
    assert result["memory"]["persisted"] is True
    bedrock.invoke_model.assert_called_once()
    payload = bedrock.invoke_model.call_args.kwargs["body"].decode()
    import json as _json
    assert _json.loads(payload)["dimensions"] == 1024
    s3vectors.put_vectors.assert_called_once()
    vector_item = s3vectors.put_vectors.call_args.kwargs["vectors"][0]
    assert len(vector_item["data"]["float32"]) == 1024
    memory.save_record.assert_called_once()
    record = memory.save_record.call_args[0][0]
    assert record.embedding_model == "amazon.titan-embed-text-v2:0"


def test_titan_failure_degrades_without_touching_vector_store():
    from datetime import datetime, timezone
    from sms_agent.memory import S3VectorStore as _Store
    mock_bedrock = MagicMock()
    mock_bedrock.invoke_model.side_effect = RuntimeError("Titan down")
    mock_s3vectors = MagicMock()
    mock_s3vectors.query_vectors.return_value = {"vectors": []}
    store = _Store(vector_bucket="vec-bucket", index_name="idx", dimension=1024,
                   s3vectors_client=mock_s3vectors)
    embed = BedrockEmbeddingProvider(bedrock_client=mock_bedrock)
    meta = FileMetadata(key="doc.txt", size_bytes=500,
                        created_at=datetime.now(timezone.utc),
                        etag='"etag-1"', bucket="bucket",
                        s3_uri="s3://bucket/doc.txt", filename="doc.txt")
    inv = MagicMock()
    inv.collect.return_value = [meta]
    rdr = MagicMock()
    rdr.get_text.return_value = RetrievedContent(
        bucket="bucket", key="doc.txt", content="financial report",
        content_type="text/plain", size_bytes=500)
    ag = MagicMock()
    ag.model_id = "test-model"
    ag.analyze_file.return_value = SemanticAnalysisResult(
        key="doc.txt", category="Finance", sensitivity="confidential",
        importance_score=0.8, confidence=0.9, reasoning="Test",
        recommended_action="retain")
    memb = MagicMock()
    memb.get_record.return_value = None
    p = SMSPipeline(bucket_name="bucket", s3_client=MagicMock(),
                    memory_store=memb, vector_store=store,
                    embedding_provider=embed)
    p.inventory, p.reader, p.agent, p.comprehend = inv, rdr, ag, MagicMock()

    result = p.process_object("doc.txt", skip_inference=False,
                              execute_action=False)
    assert result["memory"]["persisted"] is True
    mock_s3vectors.put_vectors.assert_not_called()
    memb.save_record.assert_called_once()
