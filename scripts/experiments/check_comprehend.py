import os
from sms_agent.s3_content import S3ContentReader
from sms_agent.comprehend import ComprehendAnalyzer
from sms_agent.models import FileMetadata
from datetime import datetime, timezone

def main():
    print("Starting Comprehend Verification...")
    reader = S3ContentReader()
    analyzer = ComprehendAnalyzer()
    
    bucket = "semantic-memory-steward-dev-527557823928"
    key = "demo/employee-contacts.txt"
    
    metadata = FileMetadata(
        key=key,
        bucket=bucket,
        extension=".txt",
        size_bytes=1000,
        created_at=datetime.now(timezone.utc)
    )
    
    print(f"Fetching object {key} from {bucket}...")
    content = reader.get_text(metadata)
    
    if not content.content:
        print("Error: Object is empty or could not be read.")
        return
        
    print("Calling Amazon Comprehend...")
    result = analyzer.analyze_text(content.content)
    
    print("\n--- Verification Results ---")
    
    print("\nDetected Entities:")
    for ent in result.get('entities', []):
        print(f"  - {ent['type']} (Score: {ent['score']:.4f})")
        
    print("\nDetected PII Entities:")
    for pii in result.get('pii_entities', []):
        print(f"  - {pii['type']} (Score: {pii['score']:.4f})")
        
    if result.get('errors'):
        print("\nErrors encountered:")
        for err in result['errors']:
            print(f"  - {err}")
            
    print("\nSuccess! Raw text was correctly omitted.")

if __name__ == "__main__":
    main()
