from datetime import date
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agents.analysis_request import (
    METRIC_CATALOG_VERSION,
    build_analysis_request,
    inherit_analysis_request,
    metric_catalog_payload,
)
from api.main import app
from api.portal_store import PortalStore
from api.task_runtime import SubmissionReceipt
from api.task_store import TaskStore


client = TestClient(app)


def test_unified_request_captures_metric_dimension_time_sort_and_limit():
    request = build_analysis_request("统计去年各地区实付金额 Top 10")
    assert request["metrics"] == ["paid_amount"]
    assert request["dimensions"] == ["country"]
    assert request["time_scope"] == {
        "field": "orders.order_date",
        "years": [date.today().year - 1],
        "start": None,
        "end": None,
    }
    assert request["sort"] == [{"field": "paid_amount", "direction": "desc"}]
    assert request["limit"] == 10
    assert request["metric_catalog_version"] == METRIC_CATALOG_VERSION


def test_multi_turn_inherit_replace_clear_and_unsupported_metric():
    first = build_analysis_request("统计去年各地区实付金额")
    second, country_changes = inherit_analysis_request(
        first, build_analysis_request("只看美国"), "只看美国"
    )
    assert second["metrics"] == ["paid_amount"]
    assert second["filters"]["country"] == ["United States"]
    assert any(item["field"] == "filters.country" for item in country_changes)

    third, dimension_changes = inherit_analysis_request(
        second, build_analysis_request("按月拆分"), "按月拆分"
    )
    assert third["dimensions"] == ["time"]
    assert third["time_scope"]["years"] == [date.today().year - 1]
    assert any(item["field"] == "dimensions" for item in dimension_changes)

    cleared, clear_changes = inherit_analysis_request(
        third, build_analysis_request("不限地区"), "不限地区"
    )
    assert "country" not in cleared["filters"]
    assert any(item["action"] == "clear" for item in clear_changes)

    changed, _ = inherit_analysis_request(
        third, build_analysis_request("改成净销售额"), "改成净销售额"
    )
    assert changed["metrics"] == ["net_revenue"]
    assert any("refund_amount" in item for item in changed["unsupported_conditions"])

    fresh, fresh_changes = inherit_analysis_request(
        third, build_analysis_request("新分析 客户数"), "新分析 客户数"
    )
    assert fresh["metrics"] == ["customer_count"]
    assert fresh["dimensions"] == []
    assert fresh["time_scope"]["years"] == []
    assert fresh["filters"] == {}
    assert fresh_changes == [{"field": "request", "action": "initialize"}]


def test_versioned_metric_catalog_has_required_provenance_fields():
    payload = metric_catalog_payload()
    assert payload["version"] == METRIC_CATALOG_VERSION
    paid = next(item for item in payload["metrics"] if item["metric_id"] == "paid_amount")
    assert paid["formula"] == "SUM(orders.total_amount_usd)"
    assert paid["required_fields"]
    assert paid["supported_dimensions"]
    assert paid["time_field"] == "orders.order_date"


def test_conversation_clarification_persists_and_resumes(monkeypatch, tmp_path: Path):
    store = PortalStore(tmp_path / "clarification.db")
    tasks = TaskStore(tmp_path / "clarification-tasks.db")
    settings = SimpleNamespace(
        WEB_LOGIN_ENABLED=False,
        API_AUTH_ENABLED=False,
        LOCAL_ACCESS_ENABLED=True,
        API_KEYS=(),
        API_PRINCIPALS_JSON="",
        API_RATE_LIMIT_REQUESTS=0,
    )
    monkeypatch.setattr("api.main.portal_store", store)
    monkeypatch.setattr("api.main.task_store", tasks)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)
    captured = {}

    def submit(query, intent=None, **kwargs):
        captured["query"] = query
        return SubmissionReceipt("resumed-task", replayed=False)

    monkeypatch.setattr("api.main.task_runtime.submit_with_receipt", submit)
    def resume(task_id, query, intent=None):
        captured["query"] = query
        assert tasks.resume_waiting(task_id, query, intent)
        return task_id

    monkeypatch.setattr("api.main.task_runtime.resume_waiting", resume)
    conversation = client.post("/conversations", json={"title": "clarify"}).json()
    conversation_id = conversation["conversation_id"]

    waiting = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"content": "按地区分析"},
    )
    assert waiting.status_code == 202
    assert waiting.json()["status"] == "waiting_clarification"
    waiting_task_id = waiting.json()["task_id"]
    assert tasks.get(waiting_task_id)["status"] == "waiting_clarification"
    assert "未确定分析指标" in waiting.json()["detail"]
    assert store.get_conversation(conversation_id)["pending_clarification"]

    resumed = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"content": "客户数"},
    )
    assert resumed.status_code == 202
    assert resumed.json()["status"] == "queued"
    assert resumed.json()["task_id"] == waiting_task_id
    assert tasks.get(waiting_task_id)["status"] == "queued"
    effective = resumed.json()["analysis_request"]
    assert effective["metrics"] == ["customer_count"]
    assert effective["dimensions"] == ["country"]
    assert store.get_conversation(conversation_id)["pending_clarification"] is None
    assert "已确认的结构化分析条件" in captured["query"]


def test_unsupported_metric_is_explained_without_creating_task(monkeypatch, tmp_path: Path):
    store = PortalStore(tmp_path / "unsupported.db")
    settings = SimpleNamespace(
        WEB_LOGIN_ENABLED=False, API_AUTH_ENABLED=False,
        LOCAL_ACCESS_ENABLED=True, API_KEYS=(), API_PRINCIPALS_JSON="",
        API_RATE_LIMIT_REQUESTS=0,
    )
    monkeypatch.setattr("api.main.portal_store", store)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)
    monkeypatch.setattr(
        "api.main.task_runtime.submit_with_receipt",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not submit")),
    )
    conversation_id = client.post(
        "/conversations", json={"title": "unsupported"}
    ).json()["conversation_id"]
    response = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"content": "改成净销售额"},
    )
    assert response.status_code == 202
    assert response.json()["status"] == "unsupported"
    assert "refund_amount" in response.json()["detail"]
    assert store.list_messages(conversation_id)[-1]["role"] == "system"


def test_waiting_clarification_expires_on_refresh(monkeypatch, tmp_path: Path):
    store = PortalStore(tmp_path / "expired-clarification.db")
    tasks = TaskStore(tmp_path / "expired-clarification-tasks.db")
    settings = SimpleNamespace(
        WEB_LOGIN_ENABLED=False, API_AUTH_ENABLED=False,
        LOCAL_ACCESS_ENABLED=True, API_KEYS=(), API_PRINCIPALS_JSON="",
        API_RATE_LIMIT_REQUESTS=0, CLARIFICATION_TTL_SECONDS=0,
    )
    monkeypatch.setattr("api.main.portal_store", store)
    monkeypatch.setattr("api.main.task_store", tasks)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)
    conversation_id = client.post(
        "/conversations", json={"title": "expire"}
    ).json()["conversation_id"]

    waiting = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"content": "按地区分析"},
    ).json()
    task_id = waiting["task_id"]
    refreshed = client.get(f"/conversations/{conversation_id}")

    assert refreshed.status_code == 200
    assert refreshed.json()["pending_clarification"] is None
    assert refreshed.json()["pending_task_id"] is None
    assert tasks.get(task_id)["status"] == "cancelled"
