"""Suite hermeticity: no test may leak os.environ changes.

app.py calls load_dotenv() at top level, and AppTest executes the real
app in-process — without this guard, a local D:\\SMS\\.env would leak real
keys into os.environ for the rest of the pytest process and break tests
that assert missing-key behavior (e.g. embedding provider key errors).
"""
import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _preserve_os_environ():
    with patch.dict(os.environ):
        yield
