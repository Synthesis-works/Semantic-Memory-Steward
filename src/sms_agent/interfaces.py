from typing import Protocol, List, Optional
from .models import SemanticMemoryRecord, Embedding, SemanticMatch

class EmbeddingProvider(Protocol):
    """Protocol for generating semantic embeddings."""
    
    @property
    def model_id(self) -> str:
        """Returns the unique identifier of the active embedding model."""
        ...
        
    def embed_text(self, text: str) -> List[float]:
        """Generate a vector embedding for the given text."""
        ...

class SemanticMemoryStore(Protocol):
    """Protocol for storing and retrieving document metadata."""
    
    def get_record(self, s3_uri: str) -> Optional[SemanticMemoryRecord]:
        """Fetch a semantic memory record by its S3 URI."""
        ...
        
    def save_record(self, record: SemanticMemoryRecord) -> None:
        """Persist a semantic memory record."""
        ...

class VectorStore(Protocol):
    """Protocol for persisting and querying vector embeddings."""
    
    def upsert(self, embedding: Embedding) -> None:
        """Store or update a vector embedding."""
        ...
        
    def search(self, query_vector: List[float], top_k: int = 5, threshold: float = 0.8) -> List[SemanticMatch]:
        """Find similar vectors matching the query vector above the threshold."""
        ...
