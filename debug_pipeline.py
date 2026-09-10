from sms_agent.pipeline import SMSPipeline
from sms_agent.memory import DynamoDBMemoryStore
from sms_agent.memory import S3VectorStore
import os
import traceback

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
bucket = os.getenv("SMS_S3_BUCKET", "semantic-memory-steward-dev-527557823928")
memory_store = DynamoDBMemoryStore(table_name="sms-semantic-memory")
# We also need vector_store if relationship analysis is to work properly, but it defaults to None
pipeline = SMSPipeline(bucket_name=bucket, memory_store=memory_store)

try:
    print("Processing copy...")
    result = pipeline.process_object("demo/employee-contacts-copy.txt", skip_inference=False, execute_action=False)
    print("Keys returned:", result.keys())
    if "decision" in result:
        print("Decision:", result["decision"])
except Exception as e:
    traceback.print_exc()
