import boto3
try:
    s3 = boto3.Session(profile_name='opencode', region_name='us-east-1').client('s3')
    response = s3.list_objects_v2(Bucket='semantic-memory-steward-dev-527557823928', Prefix='demo/')
    for obj in response.get('Contents', []):
        print(obj['Key'])
except Exception as e:
    print("Error:", e)
