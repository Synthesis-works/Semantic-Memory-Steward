import os
import sys

# Ensure PYTHONPATH allows importing sms_agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from sms_agent.memory import DynamoDBMemoryStore, S3VectorStore
from sms_agent.models import SemanticMemoryRecord, Embedding
from datetime import datetime, timezone
import uuid
import boto3

def run_smoke_test():
    print("Starting persistence smoke test...")
    # 1. Initialize stores
    session = boto3.Session(profile_name='opencode', region_name='us-east-1')
    
    dynamo = DynamoDBMemoryStore("sms-semantic-memory", dynamodb_resource=session.resource("dynamodb"))
    vectors = S3VectorStore(
        vector_bucket="sms-semantic-vectors-527557823928",
        index_name="sms-embeddings",
        dimension=768,
        s3vectors_client=session.client("s3vectors")
    )
    
    # 2. Create synthetic data
    # Create a completely fake document S3 URI
    test_uri = f"s3://semantic-memory-steward-dev-527557823928/smoke_test_{uuid.uuid4().hex}.txt"
    vector_id = f"vec_{uuid.uuid4().hex}"
    
    # Fake vector of size 768
    synthetic_vector = [0.01] * 768
    
    embedding = Embedding(
        vector_id=vector_id,
        vector=synthetic_vector,
        metadata={"s3_uri": test_uri, "type": "smoke_test"}
    )
    
    record = SemanticMemoryRecord(
        s3_uri=test_uri,
        bucket="semantic-memory-steward-dev-527557823928",
        key=test_uri.replace("s3://semantic-memory-steward-dev-527557823928/", ""),
        etag="fake-etag",
        size_bytes=100,
        category="misc",
        sensitivity="low",
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        embedding_model="smoke-test-model",
        content_hash="fakehash123",
        vector_id=vector_id,
        last_analyzed=datetime.now(timezone.utc),
        importance_score=0.99,
        recommended_action="retain"
    )
    
    # 3. Perform the test
    print(f"Writing vector {vector_id} to S3 Vectors...")
    vectors.upsert(embedding)
    print("Vector written successfully.")
    
    print(f"Writing record for {test_uri} to DynamoDB...")
    dynamo.save_record(record)
    print("Record written successfully.")
    
    print("Fetching record from DynamoDB...")
    fetched_record = dynamo.get_record(test_uri)
    if fetched_record:
        print("Record fetched successfully!")
        assert fetched_record.importance_score == 0.99
    else:
        print("Failed to fetch record!")
        sys.exit(1)
        
    print("Smoke test completed successfully!")

if __name__ == "__main__":
    # Ensure region and profile are picked up correctly
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    os.environ['AWS_PROFILE'] = 'opencode'
    os.environ['SMS_VECTOR_DIMENSION'] = '768'
    run_smoke_test()
