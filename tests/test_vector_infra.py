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
