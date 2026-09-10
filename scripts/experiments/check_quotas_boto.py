import boto3

client = boto3.Session(profile_name='opencode', region_name='us-east-1').client('service-quotas')
paginator = client.get_paginator('list_service_quotas')

results = []
try:
    for page in paginator.paginate(ServiceCode='sagemaker'):
        for quota in page.get('Quotas', []):
            name = quota.get('QuotaName', '')
            val = quota.get('Value', 0.0)
            if 'for endpoint usage' in name and val > 0:
                results.append((name, val))
                print(f"Found >0 quota: {name} = {val}")
except Exception as e:
    print(f"Error: {e}")

print("Done. Total >0 endpoint quotas found:", len(results))
