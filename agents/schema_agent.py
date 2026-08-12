"""
Schema Agent —— 基于 ChromaDB 语义检索 + LLM 精排，把用户业务术语映射到具体表和字段。

设计（语义 Schema 映射）：
1. ChromaDB 检索 top-k 候选字段（粗排，速度快）
2. LLM 对候选字段做精排和筛选（细排，质量高）
3. 更新 AgentState.selected_tables

这是"多智能体协同"叙事中第一个 Agent，演示了"检索 + LLM"两阶段设计。
"""

import json
import os
from agents.state import AgentState
from agents.llm import chat_with_json_output
from storage.chromadb.embedder import get_embedder
from storage.chromadb.schema_metadata import SCHEMA_FIELDS, TABLE_METADATA
from config.prompts.schema_prompt import SCHEMA_SYSTEM_PROMPT, build_schema_prompt
from agents.schema_grounding import merge_required_fields


def schema_agent_node(state: AgentState) -> AgentState:
    """
    Schema Agent 的 LangGraph 节点函数。

    输入：state.user_query
    输出：更新 state.selected_tables 和 state.current_step

    实验控制：设置环境变量 RAG_DISABLED=1 可跳过 ChromaDB 检索，
    返回全部 54 个字段（用于消融实验中的"去除RAG"版本）。
    """
    query = state["user_query"]
    state["current_step"] = "schema_agent"
    state["messages"].append("[Schema Agent] 开始语义检索...")

    # ---- 消融实验：去除 RAG（返回全部字段） ----
    if os.environ.get("RAG_DISABLED") == "1":
        all_fields = []
        for f in SCHEMA_FIELDS:
            all_fields.append({
                "table_name": f["table_name"],
                "column_name": f["column_name"],
                "dtype": f.get("dtype", ""),
                "business_term": f.get("business_term", ""),
                "relevance": "high",
            })
        state["selected_tables"] = all_fields
        state["table_context"] = _build_table_context(all_fields)
        state["messages"].append(
            f"[Schema Agent] RAG已禁用(消融实验)，返回全部 {len(all_fields)} 个字段"
        )
        return state

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
            # V4: 回填完整的 SCHEMA_FIELDS 元数据（dtype, business_term, aliases）
            enriched = merge_required_fields(
                _enrich_with_field_metadata(selected), query
            )
            state["selected_tables"] = enriched
            state["table_context"] = _build_table_context(enriched)
            state["messages"].append(
                f"[Schema Agent] LLM 精排后选中 {len(enriched)} 个字段 "
                f"(含 dtype + business_term)"
            )
            return state
    except Exception as e:
        state["messages"].append(f"[Schema Agent] LLM 调用失败，降级为 ChromaDB 粗排: {e}")

    # Fallback: 直接用 ChromaDB top-5（V4: 含 dtype + business_term）
    fallback = [
        {
            "table_name": c["table_name"],
            "column_name": c["column_name"],
            "dtype": c.get("dtype", ""),
            "business_term": c.get("business_term", ""),
            "relevance": "high" if c["rank"] <= 3 else "medium",
        }
        for c in candidates[:5]
    ]
    fallback = merge_required_fields(fallback, query)
    state["selected_tables"] = fallback
    state["table_context"] = _build_table_context(fallback)
    state["messages"].append(
        f"[Schema Agent] 降级为 ChromaDB 粗排: {len(fallback)} 个字段 "
        f"(含 dtype + business_term)"
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


def _enrich_with_field_metadata(selected: list[dict]) -> list[dict]:
    """
    V4: 对 LLM 精排后的字段列表进行元数据补充。

    从 SCHEMA_FIELDS 中查找每个字段的 dtype、business_term、aliases，
    确保 SQL Agent 拿到完整的字段元数据（不依赖 LLM 是否正确复制了这些信息）。

    Args:
        selected: LLM 返回的字段列表 [{table_name, column_name, relevance}, ...]

    Returns:
        补充了 dtype、business_term、aliases 的字段列表
    """
    # 构建 (table_name, column_name) → field_metadata 的快速查找表
    field_map = {(f["table_name"], f["column_name"]): f for f in SCHEMA_FIELDS}
    enriched = []
    for item in selected:
        key = (item["table_name"], item["column_name"])
        if key in field_map:
            f = field_map[key]
            enriched.append({
                "table_name": item["table_name"],
                "column_name": item["column_name"],
                "dtype": f["dtype"],
                "business_term": f.get("business_term", ""),
                "relevance": item.get("relevance", "medium"),
                "aliases": f.get("aliases", []),
            })
        else:
            # 在 SCHEMA_FIELDS 中未找到 — 保留原始内容
            enriched.append(item)
    return enriched


def _build_table_context(selected_tables: list[dict]) -> str:
    """
    V4: 从 TABLE_METADATA 中提取涉及表的结构化描述。

    Returns:
        表级上下文字符串，包含表描述和近似行数
    """
    table_map = {}
    for t in TABLE_METADATA:
        name = t.get("table_name", t.get("name", ""))
        if name:
            table_map[name] = t

    seen = set()
    lines = []
    for item in selected_tables:
        tname = item.get("table_name", "")
        if tname and tname not in seen and tname in table_map:
            seen.add(tname)
            tm = table_map[tname]
            desc = tm.get("description", "")
            rows = tm.get("row_count", tm.get("approx_rows", ""))
            row_str = f" (~{rows} 行)" if rows else ""
            lines.append(f"  - {tname}: {desc}{row_str}")

    if lines:
        return "涉及的表:\n" + "\n".join(lines)
    return ""
