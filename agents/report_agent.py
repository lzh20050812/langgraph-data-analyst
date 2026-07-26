"""
Report Agent —— 综合 Analysis / Prediction / Governance 的输出，用 LLM 生成经营洞察报告。

职责（LLM Agent）：
1. 接收 analysis_result + prediction_result + governance_result
2. 调用 LLM 生成结构化报告：核心发现 + 客户分析 + 运营诊断 + 预测预警 + 策略建议
3. 输出包含"为什么"（数据解读）和"怎么办"（可执行策略），不是罗列数字

这是 Phase 3 的核心文本生成 Agent，也是 LangGraph 链路中唯一直接面向最终用户的输出节点。
"""

from agents.state import AgentState
from agents.llm import chat
from config.prompts.report_prompt import REPORT_SYSTEM_PROMPT, build_report_prompt


def report_agent_node(state: AgentState) -> AgentState:
    """
    Report Agent 的 LangGraph 节点函数。

    综合所有上游 Agent 的输出，调用 LLM 生成结构化经营洞察报告，
    结果写入 state["report"]。
    """
    state["current_step"] = "report_agent"
    state["messages"].append("[Report Agent] 开始生成经营洞察报告...")

    try:
        # 收集上游输出
        analysis = state.get("analysis_result") or {}
        prediction = state.get("prediction_result") or {}
        governance = state.get("governance_result") or {}
        user_query = state.get("user_query", "")

        # 构建 prompt
        user_prompt = build_report_prompt(
            user_query=user_query,
            analysis_result=analysis,
            prediction_result=prediction,
            governance_result=governance,
        )

        messages = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 调用 LLM 生成报告（报告较长，给更大的 max_tokens）
        report_text = chat(messages, temperature=0.3, max_tokens=4096)

        if not report_text or len(report_text.strip()) < 50:
            state["error"] = "Report Agent: LLM 返回的报告过短或为空"
            state["messages"].append(f"[Report Agent] ERROR: {state['error']}")
            return state

        state["report"] = report_text.strip()
        state["messages"].append(
            f"[Report Agent] 报告生成完成 ({len(report_text)} 字符)"
        )

    except Exception as e:
        state["error"] = f"Report Agent 失败: {e}"
        state["messages"].append(f"[Report Agent] ERROR: {e}")

    return state
