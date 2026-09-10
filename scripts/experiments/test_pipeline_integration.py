import os
import json
from unittest.mock import patch
from sms_agent.pipeline import SMSPipeline
from sms_agent.models import SemanticAnalysisResult

def mock_analyze_file(self, file_key, content, metadata):
    return SemanticAnalysisResult(
        key=file_key,
        category="hr",
        sensitivity="internal",
        importance_score=0.6,
        confidence=0.9,
        reasoning="Mocked analysis before Comprehend",
        recommended_action="retain"
    )

def main():
    print("Starting pipeline integration test...")
    
    with patch("sms_agent.agent.SMSAgent.analyze_file", mock_analyze_file):
        pipeline = SMSPipeline(bucket_name="semantic-memory-steward-dev-527557823928")
        
        result = pipeline.process_object(
            file_key="demo/employee-contacts.txt",
            skip_inference=False,
            execute_action=False
        )
        
        print("\n--- Pipeline Result ---")
        print("Status:", result['status'])
        print("Analysis Sensitivity:", result['analysis']['sensitivity'])
        print("Analysis Reasoning:", result['analysis']['reasoning'])
        
        print("\nComprehend Enrichment:")
        print(json.dumps(result.get('enrichment'), indent=2))
        
        print("\nPolicy Decision Action:", result['decision']['action'])
        print("Policy Decision Reasons:", result['decision']['reasons'])

if __name__ == "__main__":
    main()
