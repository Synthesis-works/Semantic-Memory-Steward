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
    sys.modules['cfnresponse'] = mock_cfnresponse
    mock_boto3 = MockBoto3()
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
    exec(code, env)

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
