import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
from sms_agent.actions import ActionEngine
from sms_agent.models import ActionRequest, ExecutionMode

def test_ui_approval_overrides_keep():
    s3_mock = MagicMock()
    engine = ActionEngine(s3_client=s3_mock)
    req = ActionRequest(
        s3_uri="s3://bucket/demo/file.txt",
        bucket="bucket",
        key="demo/file.txt",
        requested_action="KEEP",
        reason="Human override",
        risk="LOW",
        human_approved=True
    )
    res = engine.execute(req)
    assert res.status == "VERIFIED_NO_ACTION"

def test_ui_approval_executes_quarantine():
    s3_mock = MagicMock()
    state = {"demo/file.txt": True, "trash/demo/file.txt": False}
    def mock_head(Bucket, Key):
        if not state.get(Key):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        return {}
    def mock_copy(CopySource, Bucket, Key):
        state[Key] = True
    def mock_delete(Bucket, Key):
        state[Key] = False
        
    s3_mock.head_object.side_effect = mock_head
    s3_mock.copy_object.side_effect = mock_copy
    s3_mock.delete_object.side_effect = mock_delete
    
    engine = ActionEngine(s3_client=s3_mock)
    req = ActionRequest(
        s3_uri="s3://bucket/demo/file.txt",
        bucket="bucket",
        key="demo/file.txt",
        requested_action="QUARANTINE",
        reason="Human approved",
        risk="HIGH",
        human_approved=True
    )
    res = engine.execute(req)
    assert res.status == "VERIFIED"
    assert res.action == "QUARANTINE"
    s3_mock.copy_object.assert_called_once()
    s3_mock.delete_object.assert_called_once()

def test_unapproved_review_is_blocked():
    s3_mock = MagicMock()
    engine = ActionEngine(s3_client=s3_mock)
    req = ActionRequest(
        s3_uri="s3://bucket/demo/file.txt",
        bucket="bucket",
        key="demo/file.txt",
        requested_action="QUARANTINE",
        reason="Unapproved",
        risk="HIGH",
        human_approved=False
    )
    res = engine.execute(req)
    assert res.status == "PENDING_APPROVAL"
