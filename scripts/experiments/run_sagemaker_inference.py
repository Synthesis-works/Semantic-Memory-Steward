import os
import json
from sms_agent.pipeline import SMSPipeline
import time

def run_test():
    with open("sagemaker_endpoint.json", "r") as f:
        data = json.load(f)
    endpoint = data["endpoint_name"]
    
    os.environ["SMS_LLM_PROVIDER"] = "sagemaker"
    os.environ["SMS_SAGEMAKER_ENDPOINT"] = endpoint
    os.environ["AWS_PROFILE"] = "opencode"
    
    print(f"Running inference against endpoint: {endpoint}")
    start = time.time()
    
    pipeline = SMSPipeline()
    # Execute action=False means no DynamoDB/S3 writes!
    result = pipeline.process_object("sms-semantic-memory", "demo/financial-report.txt", execute_action=False)
    
    end = time.time()
    
    print("\n--- INFERENCE RESULT ---")
    print(f"Latency: {end - start:.2f} seconds")
    print(f"Status: {result['status']}")
    
    analysis = result.get('analysis')
    if analysis:
        print(f"Category: {analysis.category}")
        print(f"Sensitivity: {analysis.sensitivity}")
        print(f"Importance Score: {analysis.importance_score}")
        print(f"Confidence: {analysis.confidence}")
        print(f"Recommended Action: {analysis.recommended_action}")
        print(f"Reasoning: {analysis.reasoning}")
    
    print(f"Comprehend Enrichment: {result.get('enrichment')}")
    print(f"Action: {result.get('action')}")
    print(f"Action Reasons: {result.get('reasons')}")

if __name__ == '__main__':
    run_test()
