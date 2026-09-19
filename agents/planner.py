"""
Planner Agent —— LangGraph 状态图入口。

职责：
1. 解析用户自然语言需求，判断意图（查数据 / 做分析 / 做预测 / 综合报告）
2. 根据意图做条件路由，决定调用哪些下游 Agent
3. 管理和传递共享状态 AgentState

Phase 3 支持的链路：
  sql_query:   START → planner → schema → sql → governance → chart → END
  analysis:    START → planner → schema → sql → governance → analysis → chart → END
  prediction:  START → planner → schema → sql → governance → prediction → chart → END
  mixed:       START → planner → schema → sql → governance → analysis → prediction → report → chart → END
"""

import json
from typing import Literal

from langgraph.graph import StateGraph, END
from agents.state import AgentState, create_initial_state
from agents.task_planning import build_task_plan, planned_nodes
from agents.observability import traced_node


# ============================================================
# 意图解析
# ============================================================

# 分析关键词（须明确请求分析/计算，而非简单查询）
ANALYSIS_KEYWORDS = [
    "rfm", "k-means", "kmeans", "聚类", "分群", "客户价值",
    "rfm分析", "客户分层", "运营指标",
    "客单价分析", "复购率分析", "品类分布",
    # V3 补充：覆盖行为分析、渠道分析、退货分析等场景
    "客户行为", "消费行为", "画像", "退货",
    # V4 补充：对比分析场景
    "对比", "比较",
]

# 预测关键词
PREDICTION_KEYWORDS = [
    "预测", "流失", "prophet", "xgboost", "趋势", "forecast",
    "未来", "销售预测", "营收预测",
]

# 报告关键词
REPORT_KEYWORDS = [
    "报告", "综合", "全面", "report", "洞察", "建议", "经营分析",
    # V3 补充：覆盖策略方案类请求
    "方案", "策略",
]

# 模糊分析信号词（query 中出现这些词但未匹配到上述关键词时，触发 LLM 再分类）
AMBIGUOUS_ANALYSIS_SIGNALS = [
    "分析", "找出", "发现", "哪些", "为什么", "怎么",
    "如何提升", "如何降低", "优化", "评估", "诊断",
]


def _get_keyword_confidence(query_lower: str) -> tuple[str, str]:
    """
    V4: 计算关键词匹配置信度。

    Returns:
        (intent, confidence_level)
        confidence_level: "high" — 明确匹配到关键词
                         "low"  — 无关键词匹配，退化为 sql_query
    """
    has_analysis = any(kw in query_lower for kw in ANALYSIS_KEYWORDS)
    has_prediction = any(kw in query_lower for kw in PREDICTION_KEYWORDS)
    has_report = any(kw in query_lower for kw in REPORT_KEYWORDS)

    # 有明确关键词 → 高置信度
    if has_report or (has_analysis and has_prediction):
        return ("mixed", "high")
    if has_analysis and not has_prediction:
        return ("analysis", "high")
    if has_prediction and not has_analysis:
        return ("prediction", "high")

    # 无关键词匹配 → 低置信度（默认 sql_query）
    return ("sql_query", "low")


def _llm_classify_intent(query: str) -> str:
    """
    V4: 使用 LLM 对模糊意图做辅助分类。

    仅在关键词无法匹配时调用（关键词退化为 sql_query 且 query 包含分析信号词）。
    要求返回 {intent: sql_query|analysis|prediction|mixed, reasoning: ...}

    Returns:
        分类后的 intent 字符串
    """
    try:
        from agents.llm import chat_with_json_output

        prompt = f"""分析以下用户查询，判断需要执行什么类型的任务。

任务类型说明:
- sql_query: 简单数据查询、统计、分组、排序，只需要SQL即可完整回答
- analysis: 需要深度数据分析（RFM客户分群、K-Means聚类、运营指标计算、对比分析等）
- prediction: 需要预测建模（客户流失预测、销售趋势预测等）
- mixed: 需要综合分析+报告输出（包含分析和建议的综合任务）

用户查询: {query}

请返回JSON: {{"intent": "<类型>", "reasoning": "<简要理由(20字内)>"}}"""

        messages = [{"role": "user", "content": prompt}]
        response = chat_with_json_output(messages, temperature=0.0, max_tokens=256)
        result = json.loads(response)
        intent = result.get("intent", "sql_query")
        if intent not in ("sql_query", "analysis", "prediction", "mixed"):
            intent = "sql_query"
        return intent
    except Exception:
        return "sql_query"


def parse_intent(state: AgentState) -> AgentState:
    """
    V4 Planner 节点：两阶段意图分类 + 日志。

    阶段1: 关键词快速匹配（覆盖 60-70% 常见case）
    阶段2: LLM 辅助分类（关键词失配时，对模糊查询做再分类）
    """
    query = state["user_query"]
    query_lower = query.lower()

    requested_intent = state.get("requested_intent")
    if requested_intent in ("sql_query", "analysis", "prediction", "mixed"):
        intent, confidence = requested_intent, "high"
        classification_method = "request_override"
    else:
        # 阶段1: 关键词置信度评估
        intent, confidence = _get_keyword_confidence(query_lower)
        classification_method = "keyword"

    # 阶段2: 低置信度 + 含分析信号词时，调用 LLM 辅助分类
    if confidence == "low" and len(query) > 10:
        # 检查是否存在模糊分析信号
        has_signal = any(sig in query_lower for sig in AMBIGUOUS_ANALYSIS_SIGNALS)
        if has_signal:
            llm_intent = _llm_classify_intent(query)
            if llm_intent != "sql_query":
                intent = llm_intent
                classification_method = "llm"

    # 生成可审计的结构化任务计划，而不是只记录粗粒度 intent。
    task_plan = build_task_plan(query, intent)
    planned_agents = planned_nodes(task_plan)

    # 写入状态
    state["intent"] = intent
    state["planner_intent"] = intent
    state["planned_agents"] = planned_agents
    state["task_plan"] = task_plan
    state["current_step"] = "intent_parsed"

    # V4 日志
    state["messages"].append(
        f"[Planner] 用户问题: {query[:100]}"
    )
    state["messages"].append(
        f"[Planner] 识别意图: {intent} (方法: {classification_method})"
    )
    state["messages"].append(
        f"[Planner] 计划调用: {' → '.join(planned_agents)}"
    )
    state["messages"].append(
        "[Planner] 任务计划: " + json.dumps(task_plan, ensure_ascii=False)
    )

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
    - sql_query → chart_renderer（结果集适合时生成通用图表）
    """
    if state.get("error"):
        return END

    task_plan = state.get("task_plan") or {}
    if task_plan.get("analysis_tools"):
        return "analysis_agent"
    if task_plan.get("prediction_tools"):
        return "prediction_agent"
    if task_plan.get("need_report"):
        return "report_agent"
    return "chart_renderer"


def route_after_analysis(state: AgentState) -> str:
    """Analysis Agent 之后的路由。"""
    if state.get("error"):
        return END

    task_plan = state.get("task_plan") or {}
    if task_plan.get("prediction_tools"):
        return "prediction_agent"
    if task_plan.get("need_report"):
        return "report_agent"
    return "chart_renderer"


def route_after_prediction(state: AgentState) -> str:
    """
    Prediction Agent 之后的路由。

    Phase 3: mixed 意图 → report_agent（生成综合报告）
             纯 prediction → chart_renderer
    """
    if state.get("error"):
        return END

    task_plan = state.get("task_plan") or {}
    if task_plan.get("need_report"):
        return "report_agent"
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
                                                 └─→ chart → END (sql_query)
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
    workflow.add_node("planner", traced_node("planner", parse_intent))
    workflow.add_node("schema_agent", traced_node("schema_agent", schema_agent_node))
    workflow.add_node("sql_agent", traced_node("sql_agent", sql_agent_node))
    workflow.add_node(
        "governance_agent", traced_node("governance_agent", governance_agent_node)
    )
    workflow.add_node("analysis_agent", traced_node("analysis_agent", analysis_agent_node))
    workflow.add_node(
        "prediction_agent", traced_node("prediction_agent", prediction_agent_node)
    )
    workflow.add_node("report_agent", traced_node("report_agent", report_agent_node))
    workflow.add_node("chart_renderer", traced_node("chart_renderer", render_charts))

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

    # Governance → Analysis / Prediction / Report / Chart
    workflow.add_conditional_edges("governance_agent", route_after_governance, {
        "analysis_agent": "analysis_agent",
        "prediction_agent": "prediction_agent",
        "report_agent": "report_agent",
        "chart_renderer": "chart_renderer",
        END: END,
    })

    # Analysis → Prediction / Chart
    workflow.add_conditional_edges("analysis_agent", route_after_analysis, {
        "prediction_agent": "prediction_agent",
        "report_agent": "report_agent",
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


def run_query(
    user_query: str,
    requested_intent: str | None = None,
    run_id: str | None = None,
) -> AgentState:
    """
    执行一次查询 —— Phase 3 的主入口。

    Args:
        user_query: 用户自然语言查询

    Returns:
        完整的 AgentState（包含 query_result, analysis_result, prediction_result,
        governance_result, report, charts）
    """
    initial_state = create_initial_state(
        user_query, requested_intent=requested_intent, run_id=run_id
    )
    from agents.scope_guard import detect_unsupported_request
    unsupported = detect_unsupported_request(user_query)
    if unsupported:
        initial_state["current_step"] = "scope_guard"
        initial_state["error"] = (
            f"当前数据能力不支持该请求（{unsupported.capability}）：{unsupported.reason}"
        )
        initial_state["messages"].append(
            f"[Scope Guard] 拒绝越界推断: {unsupported.reason}"
        )
        return initial_state
    graph = get_graph()
    result = graph.invoke(initial_state)
    return result
