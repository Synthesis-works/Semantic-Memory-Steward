import boto3
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
import json

def deploy():
    session = sagemaker.Session(boto3.Session(profile_name="opencode", region_name="us-east-1"))
    model_id = "meta-textgeneration-llama-3-8b-instruct"
    instance_type = "ml.g5.2xlarge"
    role_arn = "arn:aws:iam::527557823928:role/SageMakerExecutionRole"
    
    print(f"Deploying {model_id} on {instance_type}...")
    model = JumpStartModel(model_id=model_id, instance_type=instance_type, sagemaker_session=session, role=role_arn)
    
    # We must accept EULA for Llama 3
    predictor = model.deploy(accept_eula=True)
    
    endpoint_name = predictor.endpoint_name
    print(f"Deployment complete! Endpoint: {endpoint_name}")
    
    with open("sagemaker_endpoint.json", "w") as f:
        json.dump({"endpoint_name": endpoint_name}, f)

if __name__ == '__main__':
    deploy()
