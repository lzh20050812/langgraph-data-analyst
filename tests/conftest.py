import os


# Existing API contract tests exercise the deliberately open local mode.
# Browser-session behavior is enabled explicitly in its dedicated tests.
os.environ["WEB_LOGIN_ENABLED"] = "0"
os.environ["LOCAL_ACCESS_ENABLED"] = "1"
