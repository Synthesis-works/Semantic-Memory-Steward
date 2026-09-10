from sms_agent.pipeline import SMSPipeline
import os
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
try:
    p = SMSPipeline()
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
