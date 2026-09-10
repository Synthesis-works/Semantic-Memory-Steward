import boto3
import json
import os

def teardown():
    if not os.path.exists("sagemaker_endpoint.json"):
        print("No sagemaker_endpoint.json found. Nothing to tear down.")
        return
        
    with open("sagemaker_endpoint.json", "r") as f:
        data = json.load(f)
        
    endpoint_name = data.get("endpoint_name")
    if not endpoint_name:
        return
        
    print(f"Tearing down SageMaker endpoint: {endpoint_name}")
    client = boto3.Session(profile_name="opencode", region_name="us-east-1").client("sagemaker")
    
    try:
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleted Endpoint: {endpoint_name}")
    except Exception as e:
        print(f"Error deleting endpoint: {e}")
        
    try:
        client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Deleted EndpointConfig: {endpoint_name}")
    except Exception as e:
        print(f"Error deleting endpoint config: {e}")
        
    # The model name is usually the same or similar, we might need to list models to find it,
    # but typically JumpStart creates a model with a name similar to endpoint. 
    # Let's search for models containing the endpoint name substring or we can just leave it if it's too complex.
    # Actually, model name is often the endpoint name for JumpStart.
    try:
        # We can describe the endpoint config to get the model name
        config = client.describe_endpoint_config(EndpointConfigName=endpoint_name)
        for variant in config.get('ProductionVariants', []):
            model_name = variant['ModelName']
            client.delete_model(ModelName=model_name)
            print(f"Deleted Model: {model_name}")
    except Exception as e:
        pass

if __name__ == '__main__':
    teardown()
