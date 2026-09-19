import csv
import json
import sqlite3
from pathlib import Path

from agents.analysis_request import build_analysis_request, inherit_analysis_request
from agents.business_semantics import build_query_contract
from agents.schema_agent import _enrich_with_field_metadata
from agents.schema_agent import schema_agent_node
from agents.schema_retrieval import (
    complete_relationship_paths,
    lexical_schema_search,
)
from api.auth import requires_auth
from storage.chromadb.business_metadata import search_business_knowledge
from storage.chromadb.embedder import SchemaEmbedder
from storage.chromadb.schema_metadata import (
    SCHEMA_CATALOG_VERSION,
    get_schema_fields,
    schema_catalog_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_schema_and_business_knowledge_are_separate_and_versioned():
    fields = get_schema_fields("ai_analytics")
    business = search_business_knowledge("实付金额", data_source="ai_analytics")

    assert fields and business
    assert all(item["catalog_version"] == SCHEMA_CATALOG_VERSION for item in fields)
    assert all("column_name" in item and "formula" not in item for item in fields)
    assert business[0]["knowledge_type"] == "metric"
    assert business[0]["definition"] == "SUM(orders.total_amount_usd)"
    assert business[0]["source_id"]
    assert len(schema_catalog_fingerprint()) == 64
    assert requires_auth("/knowledge/catalog")


def test_permission_filter_happens_before_lexical_ranking():
    fields = [
        {
            "table_name": "private_sales", "column_name": "secret_margin",
            "dtype": "DECIMAL", "business_term": "秘密利润",
            "aliases": ["秘密利润"], "data_source": "ai_analytics",
            "catalog_version": "1", "source_id": "test",
            "access_scope": "tenant", "owner_id": "alice",
        },
        {
            "table_name": "public_sales", "column_name": "revenue",
            "dtype": "DECIMAL", "business_term": "公开营收",
            "aliases": ["营收"], "data_source": "ai_analytics",
            "catalog_version": "1", "source_id": "test",
            "access_scope": "public", "owner_id": "",
        },
    ]
    bob = lexical_schema_search(
        "秘密利润", schema_fields=fields, principal_id="bob"
    )
    alice = lexical_schema_search(
        "秘密利润", schema_fields=fields, principal_id="alice"
    )

    assert not any(item["column_name"] == "secret_margin" for item in bob)
    assert alice[0]["column_name"] == "secret_margin"


def test_relationship_completion_adds_required_join_keys():
    selected = [
        {"table_name": "support_tickets", "column_name": "priority"},
        {"table_name": "support_agents", "column_name": "team"},
    ]
    completed = complete_relationship_paths(selected, data_source="support_ops")
    keys = {(item["table_name"], item["column_name"]) for item in completed}

    assert ("support_tickets", "agent_id") in keys
    assert ("support_agents", "agent_id") in keys
    assert sum(item.get("relevance") == "required_join" for item in completed) == 2


def test_second_schema_isolated_and_unknown_llm_fields_are_rejected():
    support = lexical_schema_search(
        "按客服团队统计高优先级工单解决时长", data_source="support_ops"
    )
    ecommerce = lexical_schema_search(
        "按客服团队统计高优先级工单解决时长", data_source="ai_analytics"
    )

    assert support
    assert all(item["table_name"].startswith("support_") for item in support)
    assert not any(item["table_name"].startswith("support_") for item in ecommerce)
    enriched = _enrich_with_field_metadata(
        [
            {"table_name": "support_tickets", "column_name": "resolution_hours"},
            {"table_name": "support_tickets", "column_name": "invented_secret"},
        ],
        data_source="support_ops",
    )
    assert [item["column_name"] for item in enriched] == ["resolution_hours"]


def test_support_analysis_request_selects_second_data_source():
    request = build_analysis_request("按客服团队统计平均解决时长")

    assert request["data_source"] == "support_ops"
    assert request["metrics"] == ["avg_resolution_hours"]
    assert request["dimensions"] == ["team"]
    assert not request["unsupported_conditions"]

    ecommerce = build_analysis_request("按国家统计美国实付金额")
    switched, changes = inherit_analysis_request(
        ecommerce, request, "改成按客服团队统计平均解决时长"
    )
    assert switched["data_source"] == "support_ops"
    assert any(item["field"] == "data_source" for item in changes)
    assert "平均解决时长 不支持国家筛选" in switched["unsupported_conditions"]


def test_cross_source_metric_combination_is_rejected():
    request = build_analysis_request("比较实付金额和工单数")

    assert set(request["metrics"]) == {"paid_amount", "ticket_count"}
    assert "单次分析暂不支持跨数据源组合指标" in request["unsupported_conditions"]


def test_support_fixture_executes_with_different_schema():
    connection = sqlite3.connect(":memory:")
    for table in ("support_agents", "support_tickets"):
        path = ROOT / "data" / "raw" / f"{table}.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        columns = list(rows[0])
        connection.execute(
            f"CREATE TABLE {table} ({', '.join(column + ' TEXT' for column in columns)})"
        )
        connection.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})",
            [[row[column] or None for column in columns] for row in rows],
        )
    contract = build_query_contract("按客服团队统计工单平均解决时长")
    assert contract and contract.recipe_id == "support.team_resolution"
    assert build_query_contract("统计去年按客服团队的工单平均解决时长") is None
    result = connection.execute(contract.sql).fetchall()

    assert result == [("Consumer", 6, 1.36), ("Enterprise", 6, 4.5)]


def test_dense_search_applies_source_and_owner_filter_before_query():
    class Encoded:
        def tolist(self):
            return [[0.1, 0.2]]

    class Model:
        def encode(self, values):
            return Encoded()

    class Collection:
        metadata = {"catalog_fingerprint": schema_catalog_fingerprint()}

        def query(self, **kwargs):
            self.kwargs = kwargs
            return {
                "metadatas": [[{
                    "table_name": "support_tickets", "column_name": "ticket_id",
                    "business_term": "客服工单唯一标识", "dtype": "VARCHAR",
                    "data_source": "support_ops", "catalog_version": "1",
                    "source_id": "test", "access_scope": "public",
                }]],
                "documents": [["ticket"]], "distances": [[0.1]],
            }

    embedder = object.__new__(SchemaEmbedder)
    embedder._model = Model()
    embedder._collection = Collection()
    results = embedder.search(
        "工单", data_source="support_ops", principal_id="tenant-a"
    )

    where = embedder._collection.kwargs["where"]
    assert {"data_source": {"$eq": "support_ops"}} in where["$and"]
    assert {"owner_id": {"$eq": "tenant-a"}} in where["$and"][1]["$or"]
    assert results[0]["data_source"] == "support_ops"


def test_schema_agent_uses_second_source_and_separate_business_context(monkeypatch):
    class Embedder:
        def index_is_current(self):
            return True

        def search(self, *args, **kwargs):
            return []

    monkeypatch.setattr("agents.schema_agent.get_embedder", lambda: Embedder())
    monkeypatch.setattr(
        "agents.schema_agent.chat_with_json_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    query = "按客服团队统计平均解决时长"
    request = build_analysis_request(query)
    state = {
        "user_query": query, "task_plan": {"analysis_request": request},
        "messages": [], "selected_tables": [], "table_context": None,
        "business_context": None, "error": None,
    }

    result = schema_agent_node(state)
    keys = {(item["table_name"], item["column_name"]) for item in result["selected_tables"]}

    assert all(table.startswith("support_") for table, _ in keys)
    assert ("support_tickets", "agent_id") in keys
    assert ("support_agents", "agent_id") in keys
    assert "metric:avg_resolution_hours" in result["business_context"]
    assert "support_tickets.agent_id = support_agents.agent_id" in result["table_context"]


def test_second_schema_evaluation_artifact_is_scoped_and_honest():
    metrics = json.loads((
        ROOT / "evaluation" / "optimization_results"
        / "schema_multisource_20260918" / "metrics.json"
    ).read_text(encoding="utf-8"))

    assert metrics["data_source"] == "support_ops"
    assert metrics["sample_count"] == 8
    assert metrics["relationship_path_coverage"] == 1
    assert metrics["downstream_validation"]["scope"].startswith("one deterministic")
