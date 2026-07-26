"""
Schema Agent —— 基于 ChromaDB 语义检索 + LLM 精排，把用户业务术语映射到具体表和字段。

设计（对应论文"语义Schema映射"章节）：
1. ChromaDB 检索 top-k 候选字段（粗排，速度快）
2. LLM 对候选字段做精排和筛选（细排，质量高）
3. 更新 AgentState.selected_tables

这是"多智能体协同"叙事中第一个 Agent，演示了"检索 + LLM"两阶段设计。
"""

import json
from agents.state import AgentState
from agents.llm import chat_with_json_output
from storage.chromadb.embedder import get_embedder
from config.prompts.schema_prompt import SCHEMA_SYSTEM_PROMPT, build_schema_prompt


def schema_agent_node(state: AgentState) -> AgentState:
    """
    Schema Agent 的 LangGraph 节点函数。

    输入：state.user_query
    输出：更新 state.selected_tables 和 state.current_step
    """
    query = state["user_query"]
    state["current_step"] = "schema_agent"
    state["messages"].append("[Schema Agent] 开始语义检索...")

    # Step 1: ChromaDB 粗排
    embedder = get_embedder()
    if embedder.collection is None:
        state["messages"].append("[Schema Agent] ChromaDB 未初始化，尝试构建...")
        embedder.build()

    candidates = embedder.search(query, top_k=15)

    if not candidates:
        state["error"] = "Schema Agent 未检索到相关字段"
        state["messages"].append("[Schema Agent] ERROR: 未检索到相关字段")
        return state

    state["messages"].append(
        f"[Schema Agent] ChromaDB 检索到 {len(candidates)} 个候选字段"
    )

    # Step 2: LLM 精排
    # 如果 LLM 不可用（没有 API key），直接用 ChromaDB 的 top-5 结果
    try:
        messages = [
            {"role": "system", "content": SCHEMA_SYSTEM_PROMPT},
            {"role": "user", "content": build_schema_prompt(query, candidates)},
        ]
        response = chat_with_json_output(messages)

        # 解析 LLM 返回的 JSON
        selected = _parse_json_response(response)
        if selected:
            state["selected_tables"] = selected
            state["messages"].append(
                f"[Schema Agent] LLM 精排后选中 {len(selected)} 个字段"
            )
            return state
    except Exception as e:
        state["messages"].append(f"[Schema Agent] LLM 调用失败，降级为 ChromaDB 粗排: {e}")

    # Fallback: 直接用 ChromaDB top-5
    fallback = [
        {
            "table_name": c["table_name"],
            "column_name": c["column_name"],
            "relevance": "high" if c["rank"] <= 3 else "medium",
        }
        for c in candidates[:5]
    ]
    state["selected_tables"] = fallback
    state["messages"].append(
        f"[Schema Agent] 降级为 ChromaDB 粗排: {len(fallback)} 个字段"
    )

    return state


def _parse_json_response(response: str) -> list[dict] | None:
    """
    解析 LLM 返回的 JSON 响应。

    处理可能包含 markdown 代码块的情况。
    """
    text = response.strip()

    # 移除可能的 markdown 代码块标记
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and "fields" in result:
            return result["fields"]
        return None
    except json.JSONDecodeError:
        # 尝试提取 JSON 数组
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return None
