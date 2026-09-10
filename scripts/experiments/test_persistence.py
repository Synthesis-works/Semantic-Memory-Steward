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
    
    print(f"Running persistence test against endpoint: {endpoint}")
    start = time.time()
    
    pipeline = SMSPipeline()
    # Execute action=True will trigger Memory writes (DynamoDB + S3 Vectors) but no S3 mutations!
    result = pipeline.process_object("sms-semantic-memory", "demo/financial-report.txt", execute_action=True)
    
    end = time.time()
    
    print("\n--- PERSISTENCE TEST RESULT ---")
    print(f"Latency: {end - start:.2f} seconds")
    print(f"Status: {result['status']}")
    
    analysis = result.get('analysis')
    if analysis:
        print(f"Category: {analysis.category}")
        print(f"Sensitivity: {analysis.sensitivity}")
    
    print(f"Action: {result.get('action')}")
    print(f"Action Reasons: {result.get('reasons')}")

if __name__ == '__main__':
    run_test()
