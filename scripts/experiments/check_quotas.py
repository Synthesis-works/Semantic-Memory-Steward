import json
with open('sagemaker_quotas.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
quotas = data.get('Quotas', [])
for q in quotas:
    name = q.get('QuotaName', '')
    val = q.get('Value', 0.0)
    if 'for endpoint usage' in name and val > 0:
        print(f"{name}: {val}")
