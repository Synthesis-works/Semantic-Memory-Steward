import boto3
from sagemaker import session
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
import urllib.request
import json

def get_model_specs(model_id, region):
    print(f"Fetching specs for {model_id} in {region}...")
    try:
        # We can try to use JumpStartModel to just see if it initializes
        # This checks local registry of what's available
        from sagemaker.jumpstart.notebook_utils import list_jumpstart_models
        from sagemaker.jumpstart.factory.model import get_default_predictor_kwargs
        
        # We can also fetch the specs directly from the s3 bucket metadata
        bucket = get_jumpstart_content_bucket(region)
        url = f"https://{bucket}.s3.{region}.amazonaws.com/meta-textgeneration-llama-3-8b-instruct/specs/v1.1.0/specs.json"
        # The version might not be 1.1.0, let's just use sagemaker SDK APIs
        
        # Another way to get supported instances:
        from sagemaker.jumpstart.accessors import JumpStartModelsAccessor
        specs = JumpStartModelsAccessor.get_model_specs(region=region, model_id=model_id, version="*")
        
        print("\n--- Model Specifications ---")
        print("Model ID:", specs.model_id)
        print("Version:", specs.version)
        print("Default Instance Type:", specs.default_inference_instance_type)
        print("Supported Instance Types:", specs.supported_inference_instance_types)
        print("Payload Format/Predictor Class:", specs.predictor_specs.supported_content_types if hasattr(specs, 'predictor_specs') else 'Unknown')
        
    except Exception as e:
        print("Error fetching specs:", str(e))

if __name__ == '__main__':
    boto3.setup_default_session(profile_name='opencode', region_name='us-east-1')
    get_model_specs("meta-textgeneration-llama-3-8b-instruct", "us-east-1")
