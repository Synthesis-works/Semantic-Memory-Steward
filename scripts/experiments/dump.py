
        import json
        import logging
        import boto3
        import cfnresponse

        log = logging.getLogger()
        log.setLevel(logging.INFO)

        def handler(event, context):
            log.info("Received event: %s", json.dumps(event))
            request_type = event['RequestType']
            props = event['ResourceProperties']

            bucket_name = props['VectorBucketName']
            index_name = props['VectorIndexName']
            dimension = int(props['VectorDimension'])

            try:
                # Fix #1: boto3 client init inside try/except to prevent stack hang on UnknownServiceError
                client = boto3.client('s3vectors')

                if request_type == 'Create':
                    # Create bucket
                    log.info("Creating vector bucket: %s", bucket_name)
                    try:
                        client.create_vector_bucket(vectorBucketName=bucket_name)
                    except Exception as e:
                        if 'AlreadyExists' not in str(e) and 'Conflict' not in str(e):
                            raise

                    # Create index
                    log.info("Creating index %s in %s (dim=%d, metric=cosine)", index_name, bucket_name, dimension)
                    try:
                        client.create_index(
                            vectorBucketName=bucket_name,
                            indexName=index_name,
                            dataType='float32',
                            dimension=dimension,
                            distanceMetric='cosine'
                        )
                    except Exception as e:
                        # Fix #2: handle existing index for idempotency
                        if 'AlreadyExists' not in str(e) and 'Conflict' not in str(e):
                            raise
                    cfnresponse.send(event, context, cfnresponse.SUCCESS, {"Status": "Created"}, bucket_name)

                elif request_type == 'Update':
                    cfnresponse.send(event, context, cfnresponse.SUCCESS, {"Status": "Updated"}, bucket_name)

                elif request_type == 'Delete':
                    log.info("Intentionally retaining index %s and bucket %s during stack deletion", index_name, bucket_name)
                    # Explicitly doing nothing to safeguard vector data against accidental stack destruction
                    cfnresponse.send(event, context, cfnresponse.SUCCESS, {"Status": "Deleted (Retained)"}, bucket_name)

            except Exception as e:
                log.error("Failed: %s", str(e))
                cfnresponse.send(event, context, cfnresponse.FAILED, {"Message": str(e)}, bucket_name)

# -------------------------------------------------------------------------
# 4. S3 Vectors Provisioning Resource
# -------------------------------------------------------------------------
