import os
from sms_agent.models import ActionRequest, ExecutionMode
from sms_agent.actions import ActionEngine

def run():
    bucket = os.environ.get("SMS_S3_BUCKET", "semantic-memory-steward-dev-527557823928")
    source_key = "demo/financial-report-copy.txt"
    dest_key = "trash/financial-report-copy.txt"
    
    print(f"Executing manual QUARANTINE test on {bucket}")
    print(f"Source: {source_key}")
    print(f"Destination: {dest_key}")
    
    engine = ActionEngine()
    
    # 1. Unapproved Request (should fail)
    req1 = ActionRequest(
        s3_uri=f"s3://{bucket}/{source_key}",
        bucket=bucket,
        key=source_key,
        requested_action="QUARANTINE",
        reason="Testing unapproved action",
        risk="MEDIUM",
        execution_mode=ExecutionMode.SAFE,
        human_approved=False
    )
    res1 = engine.execute(req1)
    print(f"\n[Test 1] Unapproved Quarantine Result: {res1.status} - {res1.message}")
    
    # 2. Approved Request (should succeed)
    req2 = ActionRequest(
        s3_uri=f"s3://{bucket}/{source_key}",
        bucket=bucket,
        key=source_key,
        requested_action="QUARANTINE",
        reason="Testing approved action",
        risk="MEDIUM",
        execution_mode=ExecutionMode.SAFE,
        human_approved=True
    )
    res2 = engine.execute(req2)
    print(f"\n[Test 2] Approved Quarantine Result: {res2.status}")
    print(f"Message: {res2.message}")
    print(f"Action: {res2.action}")
    
    # 3. Check idempotency
    res3 = engine.execute(req2)
    print(f"\n[Test 3] Repeated Quarantine Result: {res3.status}")
    print(f"Message: {res3.message}")

if __name__ == "__main__":
    os.environ["AWS_PROFILE"] = "opencode"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    run()
