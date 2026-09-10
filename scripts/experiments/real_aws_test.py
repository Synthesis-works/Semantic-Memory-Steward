import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from sms_agent.s3_inventory import S3InventoryCollector
from sms_agent.s3_content import S3ContentReader
from sms_agent.models import FileMetadata, SemanticMemoryRecord
from sms_agent.relationships import RelationshipAnalyzer
from sms_agent.importance import ImportanceScorer
from sms_agent.policy import PolicyEngine
from sms_agent.memory import DynamoDBMemoryStore, S3VectorStore

import boto3
import hashlib
from datetime import datetime, timezone

def run_test():
    os.environ['AWS_PROFILE'] = 'opencode'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    os.environ['SMS_VECTOR_DIMENSION'] = '768'

    bucket_name = "semantic-memory-steward-dev-527557823928"
    
    session = boto3.Session(profile_name='opencode', region_name='us-east-1')
    s3_client = session.client('s3')
    dynamodb_resource = session.resource('dynamodb')
    
    print("PHASE 2 - REAL S3 INVENTORY")
    collector = S3InventoryCollector(bucket_name, prefix="demo/", s3_client=s3_client)
    objects = collector.collect()
    print(f"Found {len(objects)} objects in demo/ prefix.")
    for obj in objects:
        print(f" - {obj.key} ({obj.size_bytes} bytes, {obj.last_modified_at}) ETag: {obj.etag}")
        
    print("\nPHASE 3 - REAL CONTENT INGESTION")
    reader = S3ContentReader(s3_client=s3_client)
    
    print("\nPHASE 4 & 5 - REAL SMS ANALYSIS & DYNAMODB PERSISTENCE (NO INFERENCE)")
    dynamo_store = DynamoDBMemoryStore("sms-semantic-memory", dynamodb_resource=dynamodb_resource)
    vectors = S3VectorStore(
        vector_bucket="sms-semantic-vectors-527557823928",
        index_name="sms-embeddings",
        dimension=768,
        s3vectors_client=session.client("s3vectors") 
    )
    
    relationship_analyzer = RelationshipAnalyzer(vector_store=vectors)
    scorer = ImportanceScorer()
    policy = PolicyEngine()
    
    processed_records = []
    
    for obj in objects:
        print(f"\nProcessing {obj.key}...")
        
        # 1. Read content
        retrieved = reader.get_text(obj)
        if not retrieved or not retrieved.content:
            print(f"Skipping {obj.key}, no content")
            continue
            
        print(f"Read content: {repr(retrieved.content[:30].encode('utf-8'))}...")
        
        # 2. Hash
        hash_val = hashlib.sha256(retrieved.content.encode('utf-8')).hexdigest()
        print(f"Content hash: {hash_val}")
        
        # 3. Analyze relationships
        # S3InventoryCollector already returned a list of FileMetadata in `objects`.
        candidates = [c for c in objects if c.key != obj.key]
        
        relations = relationship_analyzer.analyze(
            target=obj,
            target_content=retrieved.content,
            candidates=candidates,
            query_vector=None 
        )
        print(f"Relationships found: {len(relations)}")
        
        # 4. Importance
        score_result = scorer.score(metadata=obj, analysis=None, relationships=relations)
        print(f"Importance Score: {score_result.score}")
        
        # 5. Policy
        from sms_agent.models import SemanticAnalysisResult
        # Provide a strictly neutral analysis result to satisfy the type checker without fabricating AI insights.
        neutral_analysis = SemanticAnalysisResult(
            key=obj.key,
            category="unclassified",
            sensitivity="internal",
            importance_score=0.0,
            confidence=0.0,
            reasoning="Neutral fallback for deterministic test.",
            recommended_action="review"
        )
        action = policy.evaluate(analysis=neutral_analysis, metadata=obj)
        print(f"Recommended Action: {action.action}")
        
        # Create Memory Record
        record = SemanticMemoryRecord(
            s3_uri=obj.s3_uri,
            bucket=obj.bucket,
            key=obj.key,
            etag=obj.etag,
            size_bytes=obj.size_bytes,
            category=neutral_analysis.category,  
            sensitivity=neutral_analysis.sensitivity,    
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            embedding_model="none", 
            content_hash=hash_val,
            vector_id="none",       
            last_analyzed=datetime.now(timezone.utc),
            importance_score=score_result.score,
            recommended_action=action.action,
        )
        
        # 6. Save to DynamoDB
        dynamo_store.save_record(record)
        print(f"Saved to DynamoDB: {record.s3_uri}")
        
        # Verify idempotency / fetch
        fetched = dynamo_store.get_record(record.s3_uri)
        if fetched:
            print(f"Verified fetch: {fetched.s3_uri} with hash {fetched.content_hash}")
        else:
            print(f"Failed to fetch {record.s3_uri}!")
            sys.exit(1)

        processed_records.append(fetched)

    print("\nIdempotency check...")
    if processed_records:
        first = processed_records[0]
        re_fetched = dynamo_store.get_record(first.s3_uri)
        print(f"Re-fetched {re_fetched.s3_uri} successfully. Hash: {re_fetched.content_hash}")
        
    print("\nPHASE 6 - S3 VECTORS CONFIGURATION")
    print(f"Vector Store instantiated with dimension: {vectors._dimension}, bucket: {vectors._bucket}, index: {vectors._index}")
    print("Vector writing skipped to avoid inventing false production embeddings.")

if __name__ == "__main__":
    run_test()
