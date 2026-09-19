import os
from pathlib import Path
import tempfile


# Existing API contract tests exercise the deliberately open local mode.
# Browser-session behavior is enabled explicitly in its dedicated tests.
os.environ["WEB_LOGIN_ENABLED"] = "0"
os.environ["LOCAL_ACCESS_ENABLED"] = "1"

# Never let tests import the process-global task/portal stores against the
# developer or Docker runtime database.  Several API tests intentionally use
# those globals, so the path must be isolated before application modules load.
_TEST_RUNTIME_DIR = tempfile.TemporaryDirectory(prefix="ai-analytics-pytest-")
os.environ["TASK_DB_PATH"] = str(Path(_TEST_RUNTIME_DIR.name) / "tasks.db")


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    _TEST_RUNTIME_DIR.cleanup()
