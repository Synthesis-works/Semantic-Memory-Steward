"""Scan-scope tests: the active workspace is discovered from S3 reality.

Active workspace = current S3 objects under the workspace prefix.
trash/ and archive/ are governance destinations, never scan targets.
Stale DynamoDB records must never masquerade as active documents.
"""
import os
from unittest.mock import MagicMock, patch

from sms_agent import ui_state
from sms_agent.ui_state import is_managed_document, is_workspace_document


def test_workspace_prefix_accepts_current_docs():
    assert is_workspace_document("demo/project-plan.txt") is True
    assert is_workspace_document("demo/any-new-file.txt") is True


def test_trash_and_archive_are_not_workspace():
    assert is_workspace_document("trash/demo/employee-contacts.txt") is False
    assert is_workspace_document("trash/financial-report-copy.txt") is False
    assert is_workspace_document("archive/demo/old-project-log.txt") is False
    assert is_workspace_document("8d31c8c241bebdcc943df587dcc35638") is False
    assert is_workspace_document("") is False


def test_managed_documents_identified():
    assert is_managed_document("trash/demo/a.txt") is True
    assert is_managed_document("archive/demo/a.txt") is True
    assert is_managed_document("demo/a.txt") is False


def test_prefix_is_configurable_not_hardcoded():
    with patch.dict(os.environ, {"SMS_WORKSPACE_PREFIX": "workspace/"}):
        assert is_workspace_document("workspace/a.txt") is True
        assert is_workspace_document("demo/a.txt") is False


def test_scope_helpers_have_no_hardcoded_demo_filenames():
    import pathlib
    text = pathlib.Path("src/sms_agent/ui_state.py").read_text(encoding="utf-8")
    for name in ("employee-contacts.txt", "financial-report.txt",
                 "old-project-log.txt", "project-plan.txt"):
        assert name not in text
