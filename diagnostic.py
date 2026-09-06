import os
import sys

# Safety configurations
os.environ["AWS_PROFILE"] = "opencode"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["SMS_BEDROCK_MODEL_ID"] = "amazon.nova-micro-v1:0"

from sms_agent.agent import SMSAgent

def run_diagnostic():
    print("=== STARTING ONE BEDROCK DIAGNOSTIC ===")
    
    agent = SMSAgent(model_id="amazon.nova-micro-v1:0")
    
    content = (
        "Project Alpha is a current internal project plan.\n"
        "The team is preparing the next milestone.\n"
        "This document contains no real personal or financial information."
    )
    
    metadata = {
        "owner": "test-user",
        "last_modified": "2026-09-06",
        "size_bytes": 1024
    }
    
    try:
        print(f"Model ID: {agent.model_id}")
        print(f"Region: {os.environ['AWS_DEFAULT_REGION']}")
        print("Invoking Bedrock using Strands Agent...")
        result = agent.analyze_file(
            file_key="test/alpha-plan.txt",
            content=content,
            metadata=metadata
        )
        print("\nSUCCESS! SemanticAnalysisResult:")
        print(result.model_dump_json(indent=2))
        
    except Exception as e:
        print("\nFAILURE!")
        print(f"Exact Exception Type: {type(e).__name__}")
        print(f"Exact AWS Error Message: {str(e)}")
        
    print("=== END DIAGNOSTIC ===")

if __name__ == "__main__":
    run_diagnostic()
