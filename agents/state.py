"""
AgentState —— 多智能体协同的运行时共享状态。

这是 LangGraph 状态图的核心数据结构，所有 Agent 通过读写 AgentState 协同工作。
注意：这是运行时状态，不是持久化知识库。一次任务执行过程中传递。

设计参考 LangGraph 的 TypedDict State 模式。
"""

from typing import TypedDict, List, Dict, Optional, Any, Annotated
import operator


class AgentState(TypedDict):
    """
    多智能体运行时共享状态。

    字段说明：
    - user_query: 用户原始自然语言输入
    - intent: Planner 解析出的意图类型（sql_query / analysis / prediction / report / mixed）
    - selected_tables: Schema Agent 识别出的相关表和字段
    - sql: SQL Agent 生成的 SQL 语句（可能多轮修正）
    - query_result: SQL 执行返回的数据行
    - sql_retries: SQL Agent 自修正重试次数
    - sql_error: 最后一次 SQL 执行错误信息
    - analysis_result: Analysis Agent 的分析结果（RFM分组/K-Means标签/运营指标）
    - prediction_result: Prediction Agent 的预测结果（流失概率/销售预测）
    - governance_result: 数据治理Agent 的质量评分
    - charts: 图表渲染数据（非Agent，由chart_renderer生成）
    - report: Report Agent 生成的最终报告文本
    - error: 全局错误信息
    - current_step: 当前执行步骤名称
    - messages: 中间消息日志（用于调试和审计）
    """

    # 输入
    user_query: str
    requested_intent: Optional[str]

    # Planner
    intent: str  # sql_query | analysis | prediction | report | mixed
    # V4: enhanced planner fields
    planner_intent: str          # LLM-classified intent (may differ from keyword-matching fallback)
    planned_agents: List[str]    # Agents planned before execution (for pipeline completeness tracking)
    task_plan: Dict[str, Any]     # Structured execution contract produced by Planner

    # Schema Agent
    selected_tables: List[Dict[str, str]]  # [{table_name, column_name, business_term, relevance}, ...]
    # V4: table-level context (descriptions, row counts — populated by Schema Agent)
    table_context: Optional[str]

    # SQL Agent
    sql: Optional[str]
    query_result: Optional[List[Dict[str, Any]]]
    sql_retries: int
    sql_retry_log: List[Dict[str, Any]]
    sql_error: Optional[str]
    evidence: Dict[str, Any]      # Traceable SQL/analysis/prediction evidence bundle

    # Analysis Agent
    analysis_result: Optional[Dict[str, Any]]  # {rfm_segments, kmeans_clusters, metrics, ...}

    # Prediction Agent
    prediction_result: Optional[Dict[str, Any]]  # {churn_model, sales_forecast, metrics, ...}

    # Governance Agent
    governance_result: Optional[Dict[str, Any]]  # {quality_score, warnings, ...}

    # Chart Renderer (非Agent)
    charts: Optional[List[Dict[str, Any]]]

    # Report Agent
    report: Optional[str]
    memory_context: List[Dict[str, Any]]

    # 错误 & 追踪
    error: Optional[str]
    current_step: str
    messages: Annotated[List[str], operator.add]  # 追加式日志


def create_initial_state(
    user_query: str, requested_intent: Optional[str] = None
) -> AgentState:
    """创建初始 AgentState（Planner 入口前调用）。"""
    return AgentState(
        user_query=user_query,
        requested_intent=requested_intent,
        intent="",
        selected_tables=[],
        table_context=None,
        sql=None,
        query_result=None,
        sql_retries=0,
        sql_retry_log=[],
        sql_error=None,
        analysis_result=None,
        prediction_result=None,
        governance_result=None,
        charts=None,
        report=None,
        memory_context=[],
        error=None,
        current_step="start",
        planner_intent="",
        planned_agents=[],
        task_plan={},
        evidence={},
        messages=[f"[Init] 收到用户查询: {user_query}"],
    )
