"""
Embedding providers for Semantic Memory Steward.

GeminiEmbeddingProvider: Temporary external fallback using gemini-embedding-2.
BedrockEmbeddingProvider: Final AWS-native provider (isolated; usable once
    the account restriction is lifted).
"""
import json
import urllib.request
import urllib.error
import os
from typing import List, Optional


class EmbeddingError(Exception):
    """Raised when an embedding provider fails."""


class GeminiEmbeddingProvider:
    """
    Temporary development embedding provider via Google Gemini API.
    Uses gemini-embedding-2 (text-embedding-004 was shut down Jan 14, 2026).
    Zero extra dependencies — reuses the existing urllib pattern.
    """

    MODEL_ID = "gemini-embedding-2"
    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError(
                "GEMINI_API_KEY is required for GeminiEmbeddingProvider"
            )

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    def embed_text(self, text: str) -> List[float]:
        """Call Gemini embedContent API and return the raw vector."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        url = f"{self.API_BASE}/{self.MODEL_ID}:embedContent?key={self._api_key}"
        payload = json.dumps({
            "model": f"models/{self.MODEL_ID}",
            "content": {"parts": [{"text": text}]},
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise EmbeddingError(
                f"Gemini embedding API error: {exc.code} - {exc.read().decode('utf-8', errors='replace')}"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc

        try:
            return body["embedding"]["values"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"Unexpected Gemini embedding response shape: {body}"
            ) from exc


class BedrockEmbeddingProvider:
    """
    AWS-native embedding provider via Amazon Bedrock (Titan Embed V2).

    Instantiation never touches the network; live calls require Bedrock
    model access and fail with a clear EmbeddingError otherwise — never
    silently returning fake data.
    """

    # Default model — Titan Text Embeddings v2, 1024 dimensions
    DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
    # Pinned output dimensionality: must match the S3 Vectors index.
    # Titan V2 supports 256/512/1024; SMS uses 1024. Never rely on the
    # service default — pin it explicitly in every request.
    OUTPUT_DIMENSION = 1024

    def __init__(self, model_id: Optional[str] = None, bedrock_client=None):
        self._model_id = model_id or os.environ.get(
            "SMS_BEDROCK_EMBEDDING_MODEL", self.DEFAULT_MODEL_ID
        )
        self._client = bedrock_client  # Accept injected client for testing

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed_text(self, text: str) -> List[float]:
        """Invoke Bedrock embeddings model and return the vector."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        import boto3  # Deferred import: boto3 already in project deps

        client = self._client or boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        payload = json.dumps({
            "inputText": text,
            "dimensions": self.OUTPUT_DIMENSION,
            "normalize": True,
        }).encode("utf-8")
        try:
            response = client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=payload,
            )
            body = json.loads(response["body"].read().decode("utf-8"))
            return body["embedding"]
        except Exception as exc:
            raise EmbeddingError(
                f"Bedrock embedding error ({self._model_id}): {exc}"
            ) from exc
