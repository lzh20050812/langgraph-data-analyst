import json
from types import SimpleNamespace
from time import sleep

from fastapi.testclient import TestClient
from langgraph.graph import END, StateGraph

from agents.business_semantics import build_query_contract
from agents.observability import traced_node
from agents.run_context import run_hooks
from agents.state import AgentState, create_initial_state
from api.auth import authenticate_api_key
from api.main import app
from api.portal_store import PortalStore
from api.result_store import result_store
from api.task_store import TaskStore
from config.prompts.report_prompt import _format_prediction_for_prompt
from agents.prediction_agent import _scope_risk_scores
from agents.task_planning import build_task_plan, deterministic_evidence_sql
import pandas as pd
from storage.chromadb.analysis_memory import AnalysisMemory
from agents.scope_guard import detect_unsupported_request


client = TestClient(app)


def test_disabled_api_key_never_authenticates_as_local_admin(monkeypatch):
    settings = SimpleNamespace(
        API_AUTH_ENABLED=False, API_KEYS=(), API_PRINCIPALS_JSON="",
        WEB_LOGIN_ENABLED=True, LOCAL_ACCESS_ENABLED=False,
    )
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)
    assert authenticate_api_key("fake-key") is None
    assert client.get(
        "/operations/workers", headers={"X-API-Key": "fake-key"}
    ).status_code == 401


def test_ownerless_legacy_result_is_admin_only(monkeypatch):
    settings = SimpleNamespace(
        API_AUTH_ENABLED=True, API_KEYS=(), WEB_LOGIN_ENABLED=False,
        LOCAL_ACCESS_ENABLED=False,
        API_PRINCIPALS_JSON=json.dumps([
            {"id": "alice", "role": "analyst", "key": "alice-key"},
            {"id": "root", "role": "admin", "key": "root-key"},
        ]),
    )
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_settings", lambda: settings)
    result_store.put("legacy-ownerless", {"report": "legacy", "charts": []})
    assert client.get(
        "/report/legacy-ownerless", headers={"X-API-Key": "alice-key"}
    ).status_code == 404
    assert client.get(
        "/report/legacy-ownerless", headers={"X-API-Key": "root-key"}
    ).status_code == 200


class _FakeModel:
    def encode(self, values):
        return SimpleNamespace(tolist=lambda: [[0.1, 0.2]])


class _FakeCollection:
    def __init__(self):
        self.where = None
        self.metadata = None

    def count(self):
        return 1

    def upsert(self, **kwargs):
        self.metadata = kwargs["metadatas"][0]

    def query(self, **kwargs):
        self.where = kwargs["where"]
        return {
            "ids": [["m1"]], "metadatas": [[{"query": "q"}]],
            "documents": [["tenant marker"]], "distances": [[0.1]],
        }


def test_analysis_memory_requires_and_filters_owner_scope():
    collection = _FakeCollection()
    memory = object.__new__(AnalysisMemory)
    memory.embedder = SimpleNamespace(
        model=_FakeModel(),
        client=SimpleNamespace(get_or_create_collection=lambda *a, **k: collection),
    )
    memory.remember(owner_id="alice", query="query", report="r" * 60)
    assert collection.metadata["owner_id"] == "alice"
    recalled = memory.recall("query", owner_id="bob")
    assert collection.where == {"owner_id": "bob"}
    assert recalled[0]["document"] == "tenant marker"


def test_real_chroma_filter_excludes_other_owner_memory():
    import chromadb

    collection = chromadb.EphemeralClient().create_collection("owner-scope")
    collection.add(
        ids=["alice", "bob"],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
        documents=["alice-only-marker", "bob-only-marker"],
        metadatas=[
            {"owner_id": "alice", "query": "q"},
            {"owner_id": "bob", "query": "q"},
        ],
    )
    memory = object.__new__(AnalysisMemory)
    memory.embedder = SimpleNamespace(
        model=_FakeModel(),
        client=SimpleNamespace(get_or_create_collection=lambda *a, **k: collection),
    )
    recalled = memory.recall("q", owner_id="bob", top_k=2)
    assert [item["document"] for item in recalled] == ["bob-only-marker"]


def test_business_error_emits_failed_and_separate_snapshot():
    events, checkpoints = [], []

    def node(state):
        state["error"] = "business failure"
        return state

    with run_hooks(
        event_sink=events.append,
        checkpoint_sink=lambda name, state: checkpoints.append(name),
    ):
        result = traced_node("demo", node)(create_initial_state("query"))
    assert events[-1]["type"] == "node_failed"
    assert result["execution_trace"][-1]["status"] == "failed"
    assert checkpoints == ["demo:failed_snapshot"]


def test_message_history_is_not_reappended_by_langgraph():
    graph = StateGraph(AgentState)
    for index in range(4):
        name = f"n{index}"

        def node(state, label=name):
            state["messages"].append(label)
            return state

        graph.add_node(name, node)
        if index:
            graph.add_edge(f"n{index - 1}", name)
    graph.set_entry_point("n0")
    graph.add_edge("n3", END)
    result = graph.compile().invoke(create_initial_state("query"))
    assert result["messages"] == ["[Init] 收到用户查询: query", "n0", "n1", "n2", "n3"]


def test_old_execution_token_cannot_write_after_reclaim(tmp_path):
    store = TaskStore(tmp_path / "ownership.db")
    store.create("task", "query", None)
    first = store.claim_next("same-worker", lease_seconds=0.01)
    old_token = first["execution_token"]
    sleep(0.02)
    assert store.recover_expired_leases(max_attempts=2)["requeued"] == 1
    second = store.claim_next("same-worker", lease_seconds=30)
    new_token = second["execution_token"]
    assert new_token != old_token
    assert store.append_event("task", {"type": "late"}, old_token) is None
    assert store.save_checkpoint("task", "late", {}, old_token) is None
    assert not store.finalize(
        "task", "completed", {"type": "late_complete"},
        result={"stale": True}, execution_token=old_token,
    )
    assert store.finalize(
        "task", "completed", {"type": "task_completed"},
        result={"fresh": True}, execution_token=new_token,
    )
    assert store.get("task")["result"] == {"fresh": True}


def test_prediction_prompt_uses_model_and_phase_not_calendar_year():
    text = _format_prediction_for_prompt({
        "churn": {"model": "LogisticRegression", "auc": 0.7},
        "sales": {
            "model": "LastValue", "rmse": 1, "mae": 1, "mape_pct": 2,
            "forecast": [
                {"ds": "2030-01-01", "yhat": 10, "phase": "backtest"},
                {"ds": "2025-01-01", "yhat": 11, "phase": "future",
                 "yhat_lower": 11, "yhat_upper": 11},
            ],
        },
    })
    assert "LogisticRegression" in text and "LastValue" in text
    assert "2025-01-01" in text and "2030-01-01" not in text
    assert "未提供有效预测区间" in text


def test_fixed_recipe_declines_unmodeled_population_filter():
    assert build_query_contract("查询不同会员等级客户数量和占比") is not None
    scoped = build_query_contract("只统计美国的会员等级客户数量和占比")
    assert "WHERE country IN ('United States')" in scoped.sql
    assert build_query_contract("排除美国的会员等级客户数量和占比") is None


def test_specialist_scope_is_carried_into_customer_and_time_evidence_sql():
    customer_plan = build_task_plan("只看美国客户的 RFM 分层", "analysis")
    customer_sql = deterministic_evidence_sql(customer_plan)
    assert customer_plan["filters"]["countries"] == ["United States"]
    assert "WHERE country IN ('United States')" in customer_sql

    sales_plan = build_task_plan("预测 2025 年的月度营收", "prediction")
    sales_sql = deterministic_evidence_sql(sales_plan)
    assert "WHERE year IN (2025)" in sales_sql


def test_unsupported_customer_period_and_region_fail_closed():
    period = detect_unsupported_request("2025年美国客户RFM分层")
    region = detect_unsupported_request("只看华东客户的会员分布")
    assert period and period.capability == "customer_historical_scope"
    assert region and region.capability == "subnational_region"


def test_churn_operational_scores_are_limited_to_current_evidence_scope():
    risk = pd.DataFrame({
        "customer_id": ["a", "b", "c"],
        "churn_probability": [0.2, 0.9, 0.8],
    })
    scoped, metadata = _scope_risk_scores(risk, {"a", "c"})
    assert set(scoped["customer_id"]) == {"a", "c"}
    assert metadata == {
        "source": "current_sql_evidence",
        "requested_count": 2,
        "matched_count": 2,
    }


def test_recent_messages_returns_tail_in_display_order(tmp_path):
    store = PortalStore(tmp_path / "portal.db")
    user = store.create_user("analyst", "StrongPass1!", "Analyst", "analyst")
    conversation = store.create_conversation(user["user_id"], "test")
    for index in range(15):
        store.add_message(conversation["conversation_id"], "user", str(index))
    recent = store.list_recent_messages(conversation["conversation_id"], limit=3)
    assert [item["content"] for item in recent] == ["12", "13", "14"]
