import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
from sms_agent.models import ActionRequest
from sms_agent.actions import ActionEngine

@pytest.fixture
def mock_s3():
    return MagicMock()

def test_keep_produces_no_mutation(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="KEEP", reason="", risk="LOW", human_approved=False)
    res = engine.execute(req)
    assert res.status == "VERIFIED_NO_ACTION"
    mock_s3.copy_object.assert_not_called()
    mock_s3.delete_object.assert_not_called()

def test_review_produces_no_mutation(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="REVIEW", reason="", risk="LOW", human_approved=False)
    res = engine.execute(req)
    assert res.status == "PENDING_APPROVAL"
    mock_s3.copy_object.assert_not_called()

def test_delete_produces_no_mutation(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="DELETE", reason="", risk="HIGH", human_approved=True)
    res = engine.execute(req)
    assert res.status == "BLOCKED"
    mock_s3.delete_object.assert_not_called()

def test_unauthorized_quarantine(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=False)
    res = engine.execute(req)
    assert res.status == "PENDING_APPROVAL"
    mock_s3.copy_object.assert_not_called()

def test_authorized_quarantine_success(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    
    state = {'trash/folder/k': False, 'folder/k': True}
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    def copy_mock(CopySource, Bucket, Key):
        state[Key] = True
        
    def delete_mock(Bucket, Key):
        state[Key] = False
        
    mock_s3.head_object.side_effect = head_mock
    mock_s3.copy_object.side_effect = copy_mock
    mock_s3.delete_object.side_effect = delete_mock
    
    req = ActionRequest(s3_uri="s3://b/folder/k", bucket="b", key="folder/k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
    res = engine.execute(req)
    assert res.status == "VERIFIED"
    assert res.action == "QUARANTINE"
    assert "trash/folder/k" in res.message
    mock_s3.copy_object.assert_called_once()

def test_copy_failure_produces_failed_result(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    state = {'trash/k': False, 'k': True}
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    mock_s3.head_object.side_effect = head_mock
    mock_s3.copy_object.side_effect = Exception("S3 Copy Error")
    
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
    res = engine.execute(req)
    assert res.status == "FAILED"
    assert "S3 Copy Error" in res.message

def test_repeated_quarantine_handled_safely(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    state = {'trash/k': True, 'k': False} # Already moved
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    mock_s3.head_object.side_effect = head_mock
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
    res = engine.execute(req)
    assert res.status == "VERIFIED"
    assert "already quarantined" in res.message
    mock_s3.copy_object.assert_not_called()

def test_quarantine_fails_if_occupied(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    state = {'trash/k': True, 'k': True} # Collision
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    mock_s3.head_object.side_effect = head_mock
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
    res = engine.execute(req)
    assert res.status == "FAILED"
    assert "already occupied" in res.message
    mock_s3.copy_object.assert_not_called()
    mock_s3.delete_object.assert_not_called()

def test_quarantine_fails_if_both_absent(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    state = {'trash/k': False, 'k': False} # Not found
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    mock_s3.head_object.side_effect = head_mock
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
    res = engine.execute(req)
    assert res.status == "FAILED"
    assert "does not exist" in res.message
    mock_s3.copy_object.assert_not_called()
    mock_s3.delete_object.assert_not_called()

def test_quarantine_rejects_unsafe_keys(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    unsafe_keys = [
        "../file.txt",
        "/demo/file.txt",
        "demo/../file.txt",
        "trash/file.txt",
        "demo//file.txt"
    ]
    for key in unsafe_keys:
        req = ActionRequest(s3_uri=f"s3://b/{key}", bucket="b", key=key, requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
        res = engine.execute(req)
        assert res.status == "FAILED"
        mock_s3.copy_object.assert_not_called()
def test_authorized_archive_success(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    
    state = {'archive/folder/k': False, 'folder/k': True}
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    def copy_mock(CopySource, Bucket, Key):
        state[Key] = True
        
    def delete_mock(Bucket, Key):
        state[Key] = False
        
    mock_s3.head_object.side_effect = head_mock
    mock_s3.copy_object.side_effect = copy_mock
    mock_s3.delete_object.side_effect = delete_mock
    
    req = ActionRequest(s3_uri="s3://b/folder/k", bucket="b", key="folder/k", requested_action="ARCHIVE", reason="", risk="LOW", human_approved=False)
    res = engine.execute(req)
    assert res.status == "VERIFIED"
    assert res.action == "ARCHIVE"
    assert "archive/folder/k" in res.message
    mock_s3.copy_object.assert_called_once()
def test_delete_failure_produces_failed_result(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    state = {'trash/k': False, 'k': True}
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    def copy_mock(CopySource, Bucket, Key):
        state[Key] = True
        
    def delete_mock(Bucket, Key):
        raise Exception("S3 Delete Error")
            
    mock_s3.head_object.side_effect = head_mock
    mock_s3.copy_object.side_effect = copy_mock
    mock_s3.delete_object.side_effect = delete_mock
    
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
    res = engine.execute(req)
    assert res.status == "FAILED"
    assert "S3 Delete Error" in res.message

def test_source_remains_produces_failed_result(mock_s3):
    engine = ActionEngine(s3_client=mock_s3)
    state = {'trash/k': False, 'k': True}
    def head_mock(Bucket, Key):
        if not state.get(Key, False):
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
            
    def copy_mock(CopySource, Bucket, Key):
        state[Key] = True
        
    def delete_mock(Bucket, Key):
        # Suppose delete "succeeds" but actually the object remains
        pass
            
    mock_s3.head_object.side_effect = head_mock
    mock_s3.copy_object.side_effect = copy_mock
    mock_s3.delete_object.side_effect = delete_mock
    
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", human_approved=True)
    res = engine.execute(req)
    assert res.status == "FAILED"
    assert "Source object was not removed" in res.message
