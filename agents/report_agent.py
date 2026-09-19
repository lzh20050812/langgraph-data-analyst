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
from config.settings import get_settings
from agents.run_context import current_owner_id
from agents.evidence import update_evidence
from agents.evidence_validation import validate_report_claims


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
        evidence = state.get("evidence") or {}
        user_query = state.get("user_query", "")
        memories = []
        owner_id = current_owner_id()
        if get_settings().ANALYSIS_MEMORY_ENABLED and owner_id:
            from storage.chromadb.analysis_memory import get_analysis_memory
            memories = get_analysis_memory().recall(
                user_query, owner_id=owner_id, top_k=3
            )
        elif get_settings().ANALYSIS_MEMORY_ENABLED:
            state["messages"].append(
                "[Report Agent] 未提供用户作用域，已禁用历史记忆复用"
            )
        state["memory_context"] = memories
        if memories:
            state["messages"].append(
                f"[Report Agent] 复用 {len(memories)} 条历史分析经验"
            )

        # 构建 prompt
        user_prompt = build_report_prompt(
            user_query=user_query,
            analysis_result=analysis,
            prediction_result=prediction,
            governance_result=governance,
            evidence=evidence,
            historical_memory=memories,
        )

        messages = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 调用 LLM 生成报告（V4: 压缩至 2000 tokens，约 1000-1500 字）
        report_text = chat(messages, temperature=0.3, max_tokens=2000)

        if not report_text or len(report_text.strip()) < 50:
            state["error"] = "Report Agent: LLM 返回的报告过短或为空"
            state["messages"].append(f"[Report Agent] ERROR: {state['error']}")
            return state

        report_text = report_text.strip()
        report_validation = validate_report_claims(report_text, evidence)
        if report_validation["status"] == "failed":
            failed_messages = [
                item["message"] for item in report_validation["checks"]
                if item["status"] == "failed"
            ]
            repair_messages = messages + [
                {"role": "assistant", "content": report_text},
                {
                    "role": "user",
                    "content": (
                        "上述报告未通过事实校验：" + "；".join(failed_messages[:8])
                        + "。请仅使用基础查询证据中的 facts 重写；每个数值必须在同一行引用对应 [F#]，"
                        "不要新增证据中不存在的数字、年份、名称或排名。"
                    ),
                },
            ]
            report_text = chat(repair_messages, temperature=0.1, max_tokens=2000).strip()
            report_validation = validate_report_claims(report_text, evidence)
        state["evidence"] = update_evidence(
            evidence, report_validation=report_validation
        )
        if report_validation["status"] == "failed":
            state["error"] = "Report Agent: 报告事实引用校验失败"
            state["messages"].append(f"[Report Agent] ERROR: {state['error']}")
            return state

        state["report"] = report_text
        if get_settings().ANALYSIS_MEMORY_ENABLED and owner_id:
            from storage.chromadb.analysis_memory import get_analysis_memory
            get_analysis_memory().remember(
                owner_id=owner_id,
                query=user_query, report=state["report"],
                category=state.get("intent", ""),
            )
        state["messages"].append(
            f"[Report Agent] 报告生成完成 ({len(report_text)} 字符)"
        )

    except Exception as e:
        state["error"] = f"Report Agent 失败: {e}"
        state["messages"].append(f"[Report Agent] ERROR: {e}")

    return state
