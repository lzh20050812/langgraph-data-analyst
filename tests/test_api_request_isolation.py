from concurrent.futures import ThreadPoolExecutor
import json
from time import sleep

from fastapi.testclient import TestClient

from api.main import app
from api.result_store import result_store


client = TestClient(app)


def _fake_run_query(query: str, requested_intent: str | None = None) -> dict:
    # Force overlapping completion order so this catches a global-latest cache.
    if query == "slow":
        sleep(0.03)
    return {
        "intent": requested_intent or "sql_query",
        "sql": "SELECT 1",
        "query_result": [{"query": query}],
        "governance_result": None,
        "analysis_result": None,
        "prediction_result": None,
        "report": f"report:{query}",
        "charts": [{"id": query}],
        "error": None,
        "messages": [],
    }


def test_concurrent_query_results_remain_request_scoped(monkeypatch):
    monkeypatch.setattr("agents.planner.run_query", _fake_run_query)
    result_store.clear()

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow_future = pool.submit(client.post, "/query", json={"query": "slow"})
        fast_future = pool.submit(client.post, "/query", json={"query": "fast"})
        slow = slow_future.result().json()
        fast = fast_future.result().json()

    assert slow["request_id"] != fast["request_id"]

    slow_report = client.get(f"/report/{slow['request_id']}")
    fast_report = client.get(f"/report/{fast['request_id']}")
    slow_charts = client.get(f"/charts/{slow['request_id']}")
    fast_charts = client.get(f"/charts/{fast['request_id']}")

    assert slow_report.json()["report"] == "report:slow"
    assert fast_report.json()["report"] == "report:fast"
    assert slow_charts.json()["charts"] == [{"id": "slow"}]
    assert fast_charts.json()["charts"] == [{"id": "fast"}]


def test_unknown_or_expired_request_id_returns_404():
    result_store.clear()

    response = client.get("/report/not-a-real-request")

    assert response.status_code == 404
    assert "不存在或已过期" in response.json()["detail"]


def test_query_input_rejects_blank_oversized_and_invalid_intent():
    assert client.post("/query", json={"query": "   "}).status_code == 422
    assert client.post("/query", json={"query": "x" * 1001}).status_code == 422
    assert client.post(
        "/query", json={"query": "客户数量", "intent": "unsupported"}
    ).status_code == 422


def test_query_body_headers_and_structured_log_share_request_id(monkeypatch):
    monkeypatch.setattr("agents.planner.run_query", _fake_run_query)
    result_store.clear()
    messages = []
    monkeypatch.setattr("api.main.logger.info", messages.append)

    response = client.post("/query", json={"query": "trace-me"})
    body = response.json()

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert len(body["request_id"]) == 32
    assert float(response.headers["X-Process-Time-Ms"]) >= 0

    records = [json.loads(message) for message in messages if message.startswith("{")]
    request_log = next(item for item in records if item.get("event") == "http_request")
    assert request_log["request_id"] == body["request_id"]
    assert request_log["path"] == "/query"
    assert request_log["status_code"] == 200
    stored = result_store.get(body["request_id"])
    assert set(stored) == {"intent", "report", "charts", "_owner_id"}
    assert stored["_owner_id"] == "local"


def test_query_result_reports_total_rows_and_truncation(monkeypatch):
    def many_rows(*args, **kwargs):
        state = _fake_run_query("many")
        state["query_result"] = [{"row": index} for index in range(105)]
        return state

    monkeypatch.setattr("agents.planner.run_query", many_rows)

    body = client.post("/query", json={"query": "many rows"}).json()

    assert len(body["query_result"]) == 100
    assert body["query_result_total_rows"] == 105
    assert body["query_result_truncated"] is True


def test_query_overload_returns_429_without_running_agents(monkeypatch):
    called = []
    monkeypatch.setattr("api.main.query_limiter.acquire", lambda: False)
    monkeypatch.setattr("agents.planner.run_query", lambda *args, **kwargs: called.append(True))

    response = client.post("/query", json={"query": "overloaded"})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"
    assert called == []


def test_query_slot_is_released_after_agent_failure(monkeypatch):
    from api.query_limiter import query_limiter

    before = query_limiter.active

    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("agents.planner.run_query", fail)
    response = client.post("/query", json={"query": "fail safely"})

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert query_limiter.active == before
