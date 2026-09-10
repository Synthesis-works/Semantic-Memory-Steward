import os
import boto3
from botocore.exceptions import ClientError
import json

def check_bedrock():
    os.environ['AWS_PROFILE'] = 'opencode'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    session = boto3.Session(profile_name='opencode', region_name='us-east-1')
    bedrock = session.client('bedrock-runtime')
    
    model_id = 'amazon.nova-micro-v1:0'
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Hello, this is a test."}]
            }
        ]
    }
    
    print(f"Testing Bedrock model: {model_id}")
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get('body').read())
        print("Bedrock is AVAILABLE.")
        print("Response:", json.dumps(response_body, indent=2))
        return True
    except ClientError as e:
        print(f"Bedrock API Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    check_bedrock()
