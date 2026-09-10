# Experiment Scripts

This directory contains one-off investigation scripts written during SMS development. They are **not production code** and are not imported by any application module.

They exist for traceability — to document what was tried, verified, or ruled out.

## Contents

| Script | Purpose |
|--------|---------|
| `check_bedrock.py` | Verify Bedrock model access status |
| `check_comprehend.py` | Manual Comprehend API test against S3 objects |
| `check_s3.py` | Verify S3 bucket contents via boto3 |
| `check_quotas_boto.py` | Enumerate SageMaker service quota values |
| `check_quotas.py` | Parse raw quota JSON output |
| `check_instances.py` | Query JumpStart supported instance types |
| `check_js.py` | JumpStart model catalogue exploration |
| `deploy_sagemaker.py` | Temporary SageMaker endpoint deployment attempt |
| `teardown_sagemaker.py` | Clean up SageMaker resources after failed deploy |
| `run_sagemaker_inference.py` | Invoke SMS pipeline against SageMaker endpoint |
| `verify_sagemaker_jumpstart.py` | Pre-deployment capability check |
| `real_aws_test.py` | Integration test against live AWS resources |
| `test_persistence.py` | Manual DynamoDB + S3 Vectors persistence test |
| `test_pipeline_integration.py` | Full pipeline integration test (with real AWS) |
| `smoke_test.py` | Quick sanity check of core imports and config |
| `diagnostic.py` | Environment and dependency diagnostics |
| `manual_quarantine.py` | Manual quarantine action test |
| `dump.py` | DynamoDB record dump utility |
| `fix.py` | One-off hotfix script |
