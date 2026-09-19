from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_remains_public_when_auth_is_enabled(monkeypatch):
    monkeypatch.setattr(
        "api.auth.get_settings",
        lambda: SimpleNamespace(API_AUTH_ENABLED=True, API_KEYS=("secret-key",)),
    )
    assert client.get("/health/live").status_code == 200


def test_analysis_routes_require_valid_key_when_enabled(monkeypatch):
    settings = SimpleNamespace(
        API_AUTH_ENABLED=True, API_KEYS=("secret-key",),
        API_PRINCIPALS_JSON="", WEB_LOGIN_ENABLED=False,
        LOCAL_ACCESS_ENABLED=False,
    )
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)

    denied = client.get("/tasks/not-found")
    assert denied.status_code == 401
    assert denied.headers["WWW-Authenticate"] == "ApiKey"

    allowed = client.get(
        "/tasks/not-found", headers={"X-API-Key": "secret-key"}
    )
    assert allowed.status_code == 404


def test_invalid_key_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "api.auth.get_settings",
        lambda: SimpleNamespace(API_AUTH_ENABLED=True, API_KEYS=("secret-key",)),
    )
    assert client.get(
        "/operations/summary", headers={"X-API-Key": "wrong"}
    ).status_code == 401


def test_readiness_rejects_enabled_auth_without_keys(monkeypatch):
    monkeypatch.setattr("storage.mysql.client.check_connection", lambda: True)
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(
            LLM_API_KEY="configured-key",
            API_AUTH_ENABLED=True,
            API_KEYS=(),
        ),
    )
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["auth"] == "not_configured"


def test_cors_preflight_allows_api_key_header():
    response = client.options(
        "/tasks",
        headers={
            "Origin": "null",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key,idempotency-key,content-type",
        },
    )
    assert response.status_code == 200
    assert "x-api-key" in response.headers["access-control-allow-headers"].lower()
    assert "idempotency-key" in response.headers["access-control-allow-headers"].lower()


def test_malformed_identity_configuration_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "api.auth.get_settings",
        lambda: SimpleNamespace(
            API_AUTH_ENABLED=True,
            API_KEYS=(),
            API_PRINCIPALS_JSON="not-json",
        ),
    )
    response = client.get(
        "/tasks/not-found", headers={"X-API-Key": "candidate"}
    )
    assert response.status_code == 503
