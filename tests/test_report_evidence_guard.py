from types import SimpleNamespace

from agents.report_agent import report_agent_node
from api.task_runtime import public_result


def _state():
    return {
        "messages": [],
        "user_query": "比较国家实付金额",
        "analysis_result": {},
        "prediction_result": {},
        "governance_result": {},
        "evidence": {
            "analysis_request": {"time_scope": {"years": [2025]}},
            "facts": [{
                "fact_id": "F1", "label": "paid_amount", "value": 120.0,
                "unit": "USD", "dimensions": {"country": "United States"},
            }],
        },
    }


def test_report_agent_repairs_uncited_numeric_claim(monkeypatch):
    responses = iter([
        "### 数据摘要\nUnited States 实付金额为 999 USD，但这里没有证据引用。" * 2,
        "### 数据摘要\n2025年 United States 实付金额为 120 USD [F1]，该结果来自当前查询。",
    ])
    monkeypatch.setattr("agents.report_agent.chat", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(
        "agents.report_agent.get_settings",
        lambda: SimpleNamespace(ANALYSIS_MEMORY_ENABLED=False),
    )

    state = report_agent_node(_state())

    assert state.get("error") is None
    assert "[F1]" in state["report"]
    assert state["evidence"]["report_validation"]["status"] == "verified"


def test_report_agent_fails_closed_after_second_invalid_report(monkeypatch):
    monkeypatch.setattr(
        "agents.report_agent.chat",
        lambda *args, **kwargs: "### 数据摘要\nUnited States 实付金额为 999 USD，缺少事实引用。" * 2,
    )
    monkeypatch.setattr(
        "agents.report_agent.get_settings",
        lambda: SimpleNamespace(ANALYSIS_MEMORY_ENABLED=False),
    )

    state = report_agent_node(_state())

    assert state["error"] == "Report Agent: 报告事实引用校验失败"
    assert "report" not in state
    assert state["evidence"]["report_validation"]["status"] == "failed"


def test_public_result_exposes_bounded_evidence_rows():
    state = {
        "run_id": "run-1", "error": None, "query_result": [],
        "evidence": {"row_count": 105, "rows": [{"v": index} for index in range(105)]},
    }
    result = public_result(state)
    assert len(result["evidence"]["rows"]) == 100
    assert result["evidence"]["rows_truncated"] is True
    assert len(state["evidence"]["rows"]) == 105
