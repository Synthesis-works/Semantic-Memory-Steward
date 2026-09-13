import yaml
import pytest

def test_template_custom_resource_delete_handler_is_safe():
    """
    Test that the inline custom resource handler in template.yaml
    does not accidentally destroy the semantic memory vector index
    or bucket when CloudFormation stack receives a Delete event.
    """
    with open("template.yaml", "r") as f:
        # Just grab the ZipFile string block.
        content = f.read()

    # Simple regex or parsing to extract the ZipFile block
    import re
    match = re.search(r'ZipFile:\s*\|(.*?)\s*# -*?\n\s*# 4\. S3 Vectors Provisioning', content, re.DOTALL)
    assert match is not None
    lines = match.group(1).split('\n')
    # Remove up to 10 spaces of indentation
    lines = [line[10:] if line.startswith(' ' * 10) else line.lstrip() for line in lines]
    code = '\n'.join(lines)

    # We will exec the code and mock boto3/cfnresponse
    class MockBoto3:
        def client(self, service):
            return MockClient()

    class MockClient:
        def delete_index(self, *args, **kwargs):
            raise RuntimeError("delete_index should not be called!")
        def delete_vector_bucket(self, *args, **kwargs):
            raise RuntimeError("delete_vector_bucket should not be called!")

    class MockCfnResponse:
        SUCCESS = "SUCCESS"
        FAILED = "FAILED"
        def send(self, event, context, status, response_data, physical_id):
            self.last_status = status
            self.last_data = response_data

    # Prepare global environment for the custom resource
    import sys
    mock_cfnresponse = MockCfnResponse()
    mock_boto3 = MockBoto3()
    prev_boto3 = sys.modules.pop('boto3', None)
    prev_cfn = sys.modules.pop('cfnresponse', None)
    sys.modules['cfnresponse'] = mock_cfnresponse
    sys.modules['boto3'] = mock_boto3

    import json
    import logging
    env = {
        'boto3': mock_boto3,
        'cfnresponse': mock_cfnresponse,
        'json': json,
        'logging': logging,
        'log': logging.getLogger('dummy'),
    }

    # Execute the ZipFile block to define `handler`
    try:
        exec(code, env)
    finally:
        if prev_boto3 is None:
            sys.modules.pop('boto3', None)
        else:
            sys.modules['boto3'] = prev_boto3
        if prev_cfn is None:
            sys.modules.pop('cfnresponse', None)
        else:
            sys.modules['cfnresponse'] = prev_cfn

    # Call the handler with a Delete event
    event = {
        'RequestType': 'Delete',
        'ResourceProperties': {
            'VectorBucketName': 'test-bucket',
            'VectorIndexName': 'test-index',
            'VectorDimension': '768'
        }
    }

    env['handler'](event, {})

    # Verify the response was SUCCESS and resources were not deleted
    assert mock_cfnresponse.last_status == "SUCCESS"
    assert "Retained" in mock_cfnresponse.last_data.get("Status", "")


def _extract_and_exec():
    """Load the inline custom-resource handler from template.yaml and exec
    it against a recording boto3/s3vectors client and a fake cfnresponse."""
    import re
    import sys
    import json
    import logging

    with open("template.yaml", "r") as f:
        content = f.read()
    match = re.search(
        r'ZipFile:\s*\|(.*?)\s*# -*?\n\s*# 4\. S3 Vectors Provisioning',
        content, re.DOTALL)
    assert match is not None
    lines = match.group(1).split('\n')
    lines = [line[10:] if line.startswith(' ' * 10) else line.lstrip()
             for line in lines]
    code = '\n'.join(lines)

    class MockCfnResponse:
        SUCCESS = "SUCCESS"
        FAILED = "FAILED"

        def __init__(self):
            self.calls = []

        def send(self, event, context, status, response_data, physical_id):
            self.calls.append((status, response_data))

    cfn = MockCfnResponse()
    sys.modules['cfnresponse'] = cfn

    class Recorder:
        def __init__(self):
            self.deletes = []
            self.creates = []
            self.delete_exc = None
            self.create_exc = None

        def client(self, service):
            self.service = service
            return self

        def delete_index(self, **kwargs):
            if self.delete_exc is not None:
                raise self.delete_exc
            self.deletes.append(kwargs)

        def create_index(self, **kwargs):
            if self.create_exc is not None:
                raise self.create_exc
            self.creates.append(kwargs)

        def create_vector_bucket(self, **kwargs):
            pass

        def delete_vector_bucket(self, **kwargs):
            pass

    rec = Recorder()
    prev_boto3 = sys.modules.pop('boto3', None)
    prev_cfn = sys.modules.pop('cfnresponse', None)
    sys.modules['boto3'] = rec
    sys.modules['cfnresponse'] = cfn
    env = {'json': json, 'logging': logging,
           'log': logging.getLogger('template-test'),
           'boto3': rec, 'cfnresponse': cfn}
    try:
        exec(code, env)
    finally:
        if prev_boto3 is None:
            sys.modules.pop('boto3', None)
        else:
            sys.modules['boto3'] = prev_boto3
        if prev_cfn is None:
            sys.modules.pop('cfnresponse', None)
        else:
            sys.modules['cfnresponse'] = prev_cfn
    return env['handler'], rec, cfn


def _update_event(old_dim, new_dim, bucket="test-bucket",
                  index="sms-embeddings"):
    return {
        'RequestType': 'Update',
        'OldResourceProperties': {
            'VectorBucketName': bucket, 'VectorIndexName': index,
            'VectorDimension': str(old_dim)},
        'ResourceProperties': {
            'VectorBucketName': bucket, 'VectorIndexName': index,
            'VectorDimension': str(new_dim)},
    }


def test_update_recreates_index_on_dimension_change():
    handler, rec, cfn = _extract_and_exec()
    handler(_update_event(768, 1024), {})
    assert rec.deletes == [{'vectorBucketName': 'test-bucket',
                            'indexName': 'sms-embeddings'}]
    assert rec.creates == [{'vectorBucketName': 'test-bucket',
                            'indexName': 'sms-embeddings',
                            'dataType': 'float32', 'dimension': 1024,
                            'distanceMetric': 'cosine'}]
    assert cfn.calls[-1][0] == "SUCCESS"
    assert "recreated" in cfn.calls[-1][1].get("Status", "").lower()


def test_update_same_dimension_is_idempotent():
    handler, rec, cfn = _extract_and_exec()
    handler(_update_event(1024, 1024), {})
    assert rec.deletes == []
    assert rec.creates == []
    assert cfn.calls[-1][0] == "SUCCESS"
    assert "no change" in cfn.calls[-1][1].get("Status", "").lower()


def test_update_tolerates_missing_old_index():
    handler, rec, cfn = _extract_and_exec()
    rec.delete_exc = Exception("NoSuchIndex: sms-embeddings does not exist")
    handler(_update_event(768, 1024), {})
    assert rec.creates
    assert cfn.calls[-1][0] == "SUCCESS"


def test_update_fails_on_delete_error():
    handler, rec, cfn = _extract_and_exec()
    rec.delete_exc = Exception("AccessDenied: not allowed")
    handler(_update_event(768, 1024), {})
    assert rec.creates == []
    assert cfn.calls[-1][0] == "FAILED"
    assert "AccessDenied" in cfn.calls[-1][1].get("Message", "")


def test_update_fails_on_create_error():
    handler, rec, cfn = _extract_and_exec()
    rec.create_exc = Exception("LimitExceededException: boom")
    handler(_update_event(768, 1024), {})
    assert cfn.calls[-1][0] == "FAILED"
    assert "LimitExceededException" in cfn.calls[-1][1].get("Message", "")
