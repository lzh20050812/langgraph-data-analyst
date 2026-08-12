import json
from pathlib import Path

from agents.business_semantics import build_query_contract, validate_result_shape
from agents.schema_grounding import invalid_qualified_columns
from agents.sql_agent import validate_select_only
from agents.sql_agent import sql_agent_node


ROOT = Path(__file__).resolve().parents[1]


def _formal_queries():
    data = json.loads((ROOT / "evaluation/data/multi_agent/test_queries.json").read_text(encoding="utf-8"))
    return data


def test_all_sql_benchmark_intents_have_reusable_semantic_contracts():
    sql_tasks = [item for item in _formal_queries() if item["task_type"] == "sql_query"]
    contracts = [build_query_contract(item["query"]) for item in sql_tasks]
    assert all(contracts)
    assert len({contract.recipe_id for contract in contracts}) == len(sql_tasks)


def test_contract_sql_is_read_only_and_schema_grounded():
    for item in _formal_queries():
        contract = build_query_contract(item["query"])
        if not contract:
            continue
        safe, reason = validate_select_only(contract.sql)
        assert safe, reason
        assert invalid_qualified_columns(contract.sql) == []


def test_high_risk_analysis_uses_safe_fallback_without_product_join():
    discount = build_query_contract("分析折扣策略与营收之间的关系，评估促销活动效果")
    returns = build_query_contract("分析退货率高的原因，制定降低退货率的策略")
    assert discount and discount.is_fallback
    assert discount.source_tables == ("orders",)
    assert returns and returns.is_fallback
    assert "o.product_id" not in returns.sql
    assert returns.source_tables == ("orders", "customers")


def test_result_shape_contract_reports_mismatch():
    contract = build_query_contract("查询不同会员等级客户数量和占比")
    assert contract
    assert validate_result_shape([{"membership_tier": "Gold", "cnt": 2, "pct": 50}], contract) is None
    issue = validate_result_shape([{"membership_tier": "Gold", "cnt": 2}], contract)
    assert issue and "结果列不符合契约" in issue


def test_held_out_business_paraphrases_map_to_same_recipes():
    cases = json.loads(
        (ROOT / "evaluation/data/sql_semantics/paraphrase_queries.json").read_text(encoding="utf-8")
    )
    for case in cases:
        contract = build_query_contract(case["query"])
        assert contract is not None, case["query"]
        assert contract.recipe_id == case["expected_recipe"], case["query"]


def test_semantic_ablation_switch_skips_contract(monkeypatch):
    from config.settings import get_settings

    settings = get_settings()
    original = settings.BUSINESS_SEMANTICS_ENABLED
    settings.BUSINESS_SEMANTICS_ENABLED = False
    monkeypatch.setattr(
        "agents.sql_agent._generate_sql",
        lambda *_: (
            "SELECT membership_tier, COUNT(*) * 100 AS pct "
            "FROM customers GROUP BY membership_tier"
        ),
    )
    monkeypatch.setattr(
        "agents.sql_agent._execute_sql",
        lambda *_: ([{"membership_tier": "Gold", "pct": 100}], None),
    )
    try:
        state = {
            "user_query": "查询不同会员等级客户数量和占比",
            "selected_tables": [{
                "table_name": "customers", "column_name": "membership_tier",
                "dtype": "VARCHAR", "business_term": "会员等级", "relevance": "required",
            }],
            "table_context": "",
            "task_plan": {},
            "messages": [],
        }
        result = sql_agent_node(state)
        assert "query_contract" not in result["task_plan"]
        assert any("业务语义层已关闭" in message for message in result["messages"])
    finally:
        settings.BUSINESS_SEMANTICS_ENABLED = original
