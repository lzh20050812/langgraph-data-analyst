"""
Planner Agent —— LangGraph 状态图入口。

职责：
1. 解析用户自然语言需求，判断意图（查数据 / 做分析 / 做预测 / 综合报告）
2. 根据意图做条件路由，决定调用哪些下游 Agent
3. 管理和传递共享状态 AgentState

Phase 3 支持的链路：
  sql_query:   START → planner → schema → sql → governance → END
  analysis:    START → planner → schema → sql → governance → analysis → chart → END
  prediction:  START → planner → schema → sql → governance → prediction → chart → END
  mixed:       START → planner → schema → sql → governance → analysis → prediction → report → chart → END
"""

import json
from typing import Literal

from langgraph.graph import StateGraph, END
from agents.state import AgentState, create_initial_state


# ============================================================
# 意图解析
# ============================================================

def parse_intent(state: AgentState) -> AgentState:
    """
    Planner 节点：解析用户意图，决定路由方向。

    使用关键词规则快速判断，覆盖主流中文业务表达。
    """
    query = state["user_query"]
    query_lower = query.lower()

    # 分析关键词（须明确请求分析/计算，而非简单查询）
    analysis_keywords = [
        "rfm", "k-means", "kmeans", "聚类", "分群", "客户价值",
        "rfm分析", "客户分层", "运营指标",
        "客单价分析", "复购率分析", "品类分布",
    ]
    # 预测关键词
    prediction_keywords = [
        "预测", "流失", "prophet", "xgboost", "趋势", "forecast",
        "未来", "销售预测", "营收预测",
    ]
    # 报告关键词
    report_keywords = ["报告", "综合", "全面", "report", "洞察", "建议", "经营分析"]

    has_analysis = any(kw in query_lower for kw in analysis_keywords)
    has_prediction = any(kw in query_lower for kw in prediction_keywords)
    has_report = any(kw in query_lower for kw in report_keywords)

    if has_report or (has_analysis and has_prediction):
        intent = "mixed"
    elif has_analysis:
        intent = "analysis"
    elif has_prediction:
        intent = "prediction"
    else:
        intent = "sql_query"

    state["intent"] = intent
    state["current_step"] = "intent_parsed"
    state["messages"].append(f"[Planner] 意图: {intent}")

    return state


# ============================================================
# 条件路由
# ============================================================

def route_after_planner(state: AgentState) -> Literal["schema_agent", END]:
    """Planner 之后：正常流程 → schema_agent；出错 → END"""
    if state.get("error"):
        return END
    return "schema_agent"


def route_after_sql(state: AgentState) -> str:
    """
    SQL Agent 执行后的路由。

    Phase 3: 所有路径都先经过 governance_agent 做数据质量评估。
    """
    if state.get("error"):
        return END

    # 所有路径都进入 governance_agent
    return "governance_agent"


def route_after_governance(state: AgentState) -> str:
    """
    数据治理 Agent 之后的路由。

    Phase 3 分支：
    - analysis → analysis_agent
    - prediction → prediction_agent
    - mixed → analysis_agent（先分析再预测）
    - sql_query → END
    """
    if state.get("error"):
        return END

    intent = state.get("intent", "sql_query")

    if intent in ("analysis", "mixed"):
        return "analysis_agent"
    elif intent == "prediction":
        return "prediction_agent"
    else:
        # sql_query: 直接结束
        return END


def route_after_analysis(state: AgentState) -> str:
    """Analysis Agent 之后的路由。"""
    if state.get("error"):
        return END

    intent = state.get("intent", "sql_query")

    if intent == "mixed":
        # 还需跑预测
        return "prediction_agent"
    else:
        # 纯 analysis → chart → END
        return "chart_renderer"


def route_after_prediction(state: AgentState) -> str:
    """
    Prediction Agent 之后的路由。

    Phase 3: mixed 意图 → report_agent（生成综合报告）
             纯 prediction → chart_renderer
    """
    if state.get("error"):
        return END

    intent = state.get("intent", "sql_query")

    if intent == "mixed":
        # 综合场景：先出报告再出图表
        return "report_agent"
    else:
        return "chart_renderer"


def route_after_report(state: AgentState) -> str:
    """Report Agent 之后的路由。"""
    if state.get("error"):
        return END
    return "chart_renderer"


# ============================================================
# 构建 LangGraph 状态图
# ============================================================

def build_graph() -> StateGraph:
    """
    构建 Phase 3 完整 LangGraph 状态图：

                        ┌───────────────────────────────────────────────────┐
                        │                                                   │
    START → planner → schema → sql → governance ─┬─→ analysis → prediction → report → chart → END
                                                 │                ↑
                                                 ├─→ prediction ──┘
                                                 │
                                                 └─→ END (sql_query)
    """
    from agents.schema_agent import schema_agent_node
    from agents.sql_agent import sql_agent_node
    from agents.analysis_agent import analysis_agent_node
    from agents.prediction_agent import prediction_agent_node
    from agents.governance_agent import governance_agent_node
    from agents.report_agent import report_agent_node
    from agents.chart_renderer import render_charts

    workflow = StateGraph(AgentState)

    # 注册所有节点
    workflow.add_node("planner", parse_intent)
    workflow.add_node("schema_agent", schema_agent_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("governance_agent", governance_agent_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("prediction_agent", prediction_agent_node)
    workflow.add_node("report_agent", report_agent_node)
    workflow.add_node("chart_renderer", render_charts)

    # 入口
    workflow.set_entry_point("planner")

    # Planner → Schema / END
    workflow.add_conditional_edges("planner", route_after_planner, {
        "schema_agent": "schema_agent",
        END: END,
    })

    # Schema → SQL
    workflow.add_edge("schema_agent", "sql_agent")

    # SQL → Governance
    workflow.add_conditional_edges("sql_agent", route_after_sql, {
        "governance_agent": "governance_agent",
        END: END,
    })

    # Governance → Analysis / Prediction / END
    workflow.add_conditional_edges("governance_agent", route_after_governance, {
        "analysis_agent": "analysis_agent",
        "prediction_agent": "prediction_agent",
        END: END,
    })

    # Analysis → Prediction / Chart
    workflow.add_conditional_edges("analysis_agent", route_after_analysis, {
        "prediction_agent": "prediction_agent",
        "chart_renderer": "chart_renderer",
        END: END,
    })

    # Prediction → Report / Chart
    workflow.add_conditional_edges("prediction_agent", route_after_prediction, {
        "report_agent": "report_agent",
        "chart_renderer": "chart_renderer",
        END: END,
    })

    # Report → Chart
    workflow.add_conditional_edges("report_agent", route_after_report, {
        "chart_renderer": "chart_renderer",
        END: END,
    })

    # Chart → END
    workflow.add_edge("chart_renderer", END)

    return workflow.compile()


# 全局编译好的图实例
_graph = None


def get_graph():
    """获取编译好的 LangGraph 图（单例）。"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(user_query: str) -> AgentState:
    """
    执行一次查询 —— Phase 3 的主入口。

    Args:
        user_query: 用户自然语言查询

    Returns:
        完整的 AgentState（包含 query_result, analysis_result, prediction_result,
        governance_result, report, charts）
    """
    graph = get_graph()
    initial_state = create_initial_state(user_query)
    result = graph.invoke(initial_state)
    return result
