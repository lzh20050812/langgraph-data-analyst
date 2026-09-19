from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from api.portal_store import PortalStore
from api.task_store import TaskStore


client = TestClient(app)


def test_passwords_sessions_and_public_user_shape(tmp_path: Path):
    store = PortalStore(tmp_path / "portal.db")
    user = store.create_user("analyst_1", "StrongPass1!", "分析员一", "analyst")
    assert "password_hash" not in user
    assert store.authenticate("analyst_1", "wrong-password") is None
    authenticated = store.authenticate("analyst_1", "StrongPass1!")
    assert authenticated["user_id"] == user["user_id"]

    token = store.create_session(user["user_id"], 3600)
    principal = store.principal_for_session(token)
    assert principal.subject == user["user_id"]
    assert principal.role == "analyst"
    assert store.revoke_session(token) is True
    assert store.principal_for_session(token) is None


def test_disabled_user_sessions_are_rejected(tmp_path: Path):
    store = PortalStore(tmp_path / "disabled.db")
    user = store.create_user("analyst_2", "StrongPass2!", "分析员二")
    token = store.create_session(user["user_id"], 3600)
    store.update_user(user["user_id"], enabled=False)
    assert store.principal_for_session(token) is None


def test_weak_passwords_are_rejected(tmp_path: Path):
    store = PortalStore(tmp_path / "password-policy.db")
    for weak_password in ("short", "alllowercase1", "ALLUPPERCASE1", "NoDigitsHere"):
        try:
            store.create_user("weak_user", weak_password, "弱密码用户")
        except ValueError as exc:
            assert "upper-case" in str(exc)
        else:
            raise AssertionError(f"weak password was accepted: {weak_password}")


def test_conversation_task_completion_adds_assistant_message(tmp_path: Path):
    store = PortalStore(tmp_path / "conversation.db")
    conversation = store.create_conversation("owner-1", "营收分析")
    message = store.add_message(
        conversation["conversation_id"], "user", "查询总营收"
    )
    store.link_task("task-1", conversation["conversation_id"], message["message_id"])
    assert store.complete_task(
        "task-1", {"report": "营收分析完成"}, None
    ) is True
    messages = store.list_messages(conversation["conversation_id"])
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[-1]["task_id"] == "task-1"


def test_conversation_can_be_renamed_and_deleted_with_children(tmp_path: Path):
    store = PortalStore(tmp_path / "conversation-management.db")
    conversation = store.create_conversation("owner-1", "旧名称")
    message = store.add_message(conversation["conversation_id"], "user", "测试消息")
    store.link_task("task-delete", conversation["conversation_id"], message["message_id"])

    updated = store.update_conversation(conversation["conversation_id"], "新名称")
    assert updated["title"] == "新名称"
    assert store.delete_conversation(conversation["conversation_id"]) is True
    assert store.get_conversation(conversation["conversation_id"]) is None
    assert store.list_messages(conversation["conversation_id"]) == []


def test_cached_demo_endpoint_persists_labelled_task(monkeypatch, tmp_path: Path):
    portal = PortalStore(tmp_path / "demo-portal.db")
    tasks = TaskStore(tmp_path / "demo-tasks.db")
    settings = SimpleNamespace(
        WEB_LOGIN_ENABLED=False,
        API_AUTH_ENABLED=False,
        API_KEYS=(),
        API_PRINCIPALS_JSON="",
        DEMO_MODE_ENABLED=True,
        AUDIT_RETENTION_SECONDS=3600,
        AUDIT_MAX_RECORDS=100,
    )
    monkeypatch.setattr("api.main.portal_store", portal)
    monkeypatch.setattr("api.main.task_store", tasks)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)

    conversation = client.post("/conversations", json={"title": "演示"})
    assert conversation.status_code == 201
    conversation_id = conversation.json()["conversation_id"]
    response = client.post(f"/conversations/{conversation_id}/demo/category-performance")
    assert response.status_code == 201
    task = tasks.get(response.json()["task_id"])
    assert task["status"] == "completed"
    assert task["result"]["demo_mode"] is True
    assert "不调用 LLM" in task["result"]["source_notice"]
    assert task["result"]["charts"]

    recovery = client.post(
        f"/conversations/{conversation_id}/demo/worker-recovery"
    )
    assert recovery.status_code == 201
    recovery_id = recovery.json()["task_id"]
    recovery_task = tasks.get(recovery_id)
    assert recovery_task["status"] == "completed"
    assert recovery_task["result"]["demo_kind"] == "failure_recovery"
    assert [event["type"] for event in tasks.events_after(recovery_id)] == [
        "task_queued",
        "task_claimed",
        "task_requeued",
        "task_claimed",
        "task_started",
        "task_completed",
    ]


def test_cookie_login_and_logout(monkeypatch, tmp_path: Path):
    store = PortalStore(tmp_path / "login.db")
    user = store.create_user("portal_admin", "StrongPass3!", "门户管理员", "admin")
    settings = SimpleNamespace(
        WEB_LOGIN_ENABLED=True,
        API_AUTH_ENABLED=False,
        SESSION_TTL_SECONDS=3600,
        SESSION_COOKIE_SECURE=False,
    )
    monkeypatch.setattr("api.main.portal_store", store)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)
    monkeypatch.setattr(
        "api.main.task_store.consume_rate_limit",
        lambda **_: {
            "allowed": True, "limit": 5, "remaining": 4,
            "reset_at": 300, "retry_after": 1,
        },
    )
    monkeypatch.setattr("api.main.task_store.append_audit", lambda **_: 1)

    response = client.post(
        "/auth/login",
        json={"username": "portal_admin", "password": "StrongPass3!"},
    )
    assert response.status_code == 200
    assert response.json()["user"]["user_id"] == user["user_id"]
    assert "HttpOnly" in response.headers["set-cookie"]
    identity = client.get("/auth/me")
    assert identity.status_code == 200
    assert identity.json()["user"]["display_name"] == "门户管理员"
    logout = client.post("/auth/logout")
    assert logout.status_code == 200
    assert client.get("/auth/me").status_code == 401
