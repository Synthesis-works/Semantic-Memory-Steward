import pytest
from sms_agent.models import ActionRequest, ExecutionMode
from sms_agent.authorization import ActionAuthorizer

def test_safe_quarantine_no_approval():
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", execution_mode=ExecutionMode.SAFE, human_approved=False)
    auth = ActionAuthorizer()
    assert auth.validate(req) == "PENDING_APPROVAL"

def test_safe_quarantine_with_approval():
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="MEDIUM", execution_mode=ExecutionMode.SAFE, human_approved=True)
    auth = ActionAuthorizer()
    assert auth.validate(req) == "AUTHORIZED"

def test_safe_delete_with_approval_is_blocked():
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="DELETE", reason="", risk="HIGH", execution_mode=ExecutionMode.SAFE, human_approved=True)
    auth = ActionAuthorizer()
    assert auth.validate(req) == "BLOCKED"

def test_review_requires_approval():
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="REVIEW", reason="", risk="LOW", execution_mode=ExecutionMode.SAFE, human_approved=True)
    auth = ActionAuthorizer()
    assert auth.validate(req) == "PENDING_APPROVAL"

def test_keep_allowed_as_noop():
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="KEEP", reason="", risk="LOW", execution_mode=ExecutionMode.SAFE, human_approved=False)
    auth = ActionAuthorizer()
    assert auth.validate(req) == "AUTHORIZED"

def test_invalid_action_rejected():
    # pydantic will validate the literal string, so we construct without validation or test something invalid but caught by authorizer
    # Actually, we can just bypass pydantic validation for testing if needed
    req = ActionRequest.model_construct(s3_uri="s3://b/k", bucket="b", key="k", requested_action="INVALID", reason="", risk="LOW", execution_mode=ExecutionMode.SAFE, human_approved=True)
    auth = ActionAuthorizer()
    assert auth.validate(req) == "BLOCKED"

def test_high_risk_cannot_bypass():
    req = ActionRequest(s3_uri="s3://b/k", bucket="b", key="k", requested_action="QUARANTINE", reason="", risk="HIGH", execution_mode=ExecutionMode.AUTONOMOUS, human_approved=False)
    auth = ActionAuthorizer()
    assert auth.validate(req) == "BLOCKED"
