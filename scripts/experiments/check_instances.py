import boto3
import sagemaker
from sagemaker.jumpstart import utils

boto_sess = boto3.Session(profile_name='opencode', region_name='us-east-1')
sm_sess = sagemaker.Session(boto_session=boto_sess)

print("LLama 3 8B Instruct:")
try:
    print(utils.get_supported_inference_instance_types("meta-textgeneration-llama-3-8b-instruct", region="us-east-1", sagemaker_session=sm_sess))
except Exception as e:
    print(e)

print("\nGPT-2:")
try:
    print(utils.get_supported_inference_instance_types("huggingface-textgeneration-gpt2", region="us-east-1", sagemaker_session=sm_sess))
except Exception as e:
    print(e)
