from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_liveness_does_not_require_external_dependencies():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_local_file_dashboard_is_allowed_by_cors():
    response = client.options(
        "/health",
        headers={
            "Origin": "null",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "null"


def test_readiness_reports_configured_dependencies(monkeypatch):
    monkeypatch.setattr("storage.mysql.client.check_connection", lambda: True)
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(LLM_API_KEY="configured-key"),
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {
        "mysql": "connected",
        "llm": "configured",
    }


def test_readiness_returns_503_when_a_dependency_is_not_ready(monkeypatch):
    monkeypatch.setattr("storage.mysql.client.check_connection", lambda: False)
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(LLM_API_KEY=""),
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_external_mode_readiness_requires_an_active_worker(monkeypatch):
    monkeypatch.setattr("storage.mysql.client.check_connection", lambda: True)
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(
            LLM_API_KEY="configured-key",
            TASK_EXECUTION_MODE="external",
            TASK_WORKER_LEASE_SECONDS=60,
        ),
    )
    monkeypatch.setattr("api.main.task_store.list_workers", lambda **_: [])

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["worker"] == "unavailable"


def test_external_mode_readiness_accepts_an_active_worker(monkeypatch):
    monkeypatch.setattr("storage.mysql.client.check_connection", lambda: True)
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(
            LLM_API_KEY="configured-key",
            TASK_EXECUTION_MODE="external",
            TASK_WORKER_LEASE_SECONDS=60,
        ),
    )
    monkeypatch.setattr(
        "api.main.task_store.list_workers",
        lambda **_: [{"worker_id": "worker-a", "status": "idle"}],
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["worker"] == "connected"


def test_readiness_rejects_invalid_rate_limit_configuration(monkeypatch):
    monkeypatch.setattr("storage.mysql.client.check_connection", lambda: True)
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(
            LLM_API_KEY="configured-key",
            API_RATE_LIMIT_REQUESTS=-1,
            API_RATE_LIMIT_WINDOW_SECONDS=0,
        ),
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["rate_limit"] == "invalid_configuration"


def test_llm_client_uses_bounded_timeout_and_retries(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("agents.llm.OpenAI", fake_openai)
    monkeypatch.setattr(
        "agents.llm.get_settings",
        lambda: SimpleNamespace(
            LLM_API_KEY="key",
            LLM_API_BASE="https://llm.example/v1",
            LLM_TIMEOUT_SECONDS=17.5,
            LLM_MAX_RETRIES=3,
        ),
    )

    from agents.llm import get_llm_client

    get_llm_client()

    assert captured["timeout"] == 17.5
    assert captured["max_retries"] == 3


def test_mysql_engine_enables_stale_connection_detection(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return sentinel

    settings = SimpleNamespace(
        mysql_url="mysql+pymysql://reader:password@mysql/db",
        MYSQL_POOL_TIMEOUT_SECONDS=9,
        MYSQL_CONNECT_TIMEOUT_SECONDS=4,
    )
    monkeypatch.setattr("storage.mysql.client._engine", None)
    monkeypatch.setattr("storage.mysql.client.get_settings", lambda: settings)
    monkeypatch.setattr("storage.mysql.client.create_engine", fake_create_engine)

    from storage.mysql.client import get_engine

    assert get_engine() is sentinel
    assert captured["pool_pre_ping"] is True
    assert captured["pool_timeout"] == 9
    assert captured["connect_args"] == {"connect_timeout": 4}
