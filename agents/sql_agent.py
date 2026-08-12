"""
SQL Agent —— 自然语言 → SQL 生成 → 执行 → 自修正闭环。

核心设计（LLM SQL 生成与自修正）：
1. 根据 Schema Agent 选中的表/字段信息生成 SQL
2. SELECT-only 安全校验（防御 LLM 生成 DROP/DELETE 等危险语句）
3. 在 MySQL 上真实执行
4. 若执行失败 → 把报错信息喂回 LLM 重新生成（最多3次重试）
5. 详细记录每次重试信息，供 eval_sql.py 画出"重试次数 vs 成功率"曲线
"""

import re
import sqlparse
from agents.state import AgentState
from agents.evidence import build_evidence_bundle
from agents.business_semantics import build_query_contract, validate_result_shape
from agents.schema_grounding import (
    invalid_qualified_columns,
    missing_required_columns,
    percentage_scale_issue,
)
from agents.task_planning import deterministic_evidence_sql
from agents.llm import chat
from config.prompts.sql_prompt import (
    SQL_SYSTEM_PROMPT,
    build_sql_prompt,
    build_sql_retry_prompt,
)
from config.settings import get_settings


# ============================================================
# SQL 安全校验
# ============================================================

# 危险 SQL 关键词（拒绝执行）
DANGEROUS_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
    "TRUNCATE", "RENAME", "REPLACE", "GRANT", "REVOKE",
    "EXEC", "EXECUTE", "CALL", "LOAD", "INTO OUTFILE", "INTO DUMPFILE",
]


def validate_select_only(sql: str) -> tuple[bool, str]:
    """
    校验 SQL 是否为安全的 SELECT 语句。

    使用 sqlparse 解析 + 正则双重校验，确保不会执行危险操作。

    Returns:
        (is_safe, reason)
    """
    if not sql or not sql.strip():
        return False, "SQL 为空"

    sql_clean = sql.strip()

    # 方法1: sqlparse 解析语句类型
    try:
        parsed = sqlparse.parse(sql_clean)
        if parsed:
            stmt_type = parsed[0].get_type()
            if stmt_type != "SELECT" and stmt_type != "UNKNOWN":
                return False, f"语句类型为 {stmt_type}，仅允许 SELECT"
    except Exception:
        pass

    # 方法2: 关键词正则检查（双重保险）
    sql_upper = sql_clean.upper()
    for keyword in DANGEROUS_KEYWORDS:
        # 用单词边界匹配，避免误判（如字段名中有 order 不会被匹配）
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper):
            return False, f"检测到危险关键词: {keyword}"

    # 必须以 SELECT 开头（忽略前导空格和注释）
    # 移除单行注释
    sql_no_comments = re.sub(r'--[^\n]*', '', sql_upper)
    # 移除多行注释
    sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)
    sql_no_comments = sql_no_comments.strip()

    if not sql_no_comments.startswith("SELECT") and not sql_no_comments.startswith("WITH"):
        return False, f"SQL 必须以 SELECT 或 WITH 开头，实际: {sql_no_comments[:50]}..."

    return True, "OK"


# ============================================================
# SQL 生成
# ============================================================

def _build_schema_info_str(selected_tables: list[dict], table_context: str = None) -> str:
    """
    V4: 将 Schema Agent 选中的字段列表格式化为 LLM 可读的增强文本。

    输出格式:
      [表名] — 表描述（~行数）
        table.column (数据类型): 业务含义 [相关度: high/medium]
    """
    lines = []

    # 表级上下文（描述、行数、关系）
    if table_context:
        lines.append(f"=== 表级信息 ===\n{table_context}\n")

    lines.append("=== 字段详细信息 ===")
    seen_tables = set()
    for item in selected_tables:
        t = item.get("table_name", "")
        if t not in seen_tables:
            lines.append(f"\n[{t}]")
            seen_tables.add(t)

        col = item.get("column_name", "?")
        dtype = item.get("dtype", "")
        business_term = item.get("business_term", "")
        relevance = item.get("relevance", "")

        # 主行: table.column (TYPE)
        dtype_str = f" ({dtype})" if dtype else ""
        lines.append(f"  {t}.{col}{dtype_str}")

        # 业务含义
        if business_term:
            lines.append(f"    业务含义: {business_term}")

        # 相关度
        if relevance:
            lines.append(f"    相关度: {relevance}")

    return "\n".join(lines)


def _generate_sql(user_query: str, schema_info: str) -> str:
    """调用 LLM 生成 SQL。"""
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": build_sql_prompt(user_query, schema_info)},
    ]
    response = chat(messages)
    return _clean_sql_response(response)


def _retry_sql(
    user_query: str, failed_sql: str, error_msg: str, schema_info: str
) -> str:
    """将错误信息喂回 LLM，要求修正 SQL。"""
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_sql_retry_prompt(
                user_query, failed_sql, error_msg, schema_info
            ),
        },
    ]
    response = chat(messages)
    return _clean_sql_response(response)


def _clean_sql_response(response: str) -> str:
    """清理 LLM 返回的 SQL：去掉 markdown 代码块、前后空白。"""
    text = response.strip()

    # 去掉 ```sql ... ``` 包裹
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行 ```sql 和末行 ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 如果 LLM 在前面加了说明文字，尝试提取 SQL
    # 查找 SELECT 关键字位置
    select_pos = text.upper().find("SELECT")
    with_pos = text.upper().find("WITH")
    if with_pos >= 0 and (with_pos < select_pos or select_pos < 0):
        start_pos = with_pos
    elif select_pos >= 0:
        start_pos = select_pos
    else:
        start_pos = 0

    if start_pos > 0:
        text = text[start_pos:]

    return text.strip()


# ============================================================
# SQL 执行
# ============================================================

def _execute_sql(sql: str, adapter=None) -> tuple[list[dict] | None, str | None]:
    """
    在数据库上执行 SQL（自动选择 MySQL 或 DuckDB）。

    Args:
        sql: SQL 语句
        adapter: 可选的 DBAdapter 实例，不传则自动检测

    Returns:
        (rows, error): 成功时 error=None；失败时 rows=None
    """
    try:
        if adapter is None:
            from storage.db_adapter import get_available_adapter
            adapter = get_available_adapter()
        rows = adapter.execute_sql(sql)
        return rows, None
    except Exception as e:
        return None, str(e)


# ============================================================
# LangGraph 节点
# ============================================================

def sql_agent_node(state: AgentState) -> AgentState:
    """
    SQL Agent 的 LangGraph 节点函数。

    执行流程：
    1. 根据 selected_tables 生成 SQL
    2. 安全校验 → 不通过则拒绝执行
    3. 执行 SQL
    4. 若失败 → 重试（最多3次），每次记录详情
    5. 更新 state.query_result / state.sql / state.sql_retries
    """
    settings = get_settings()
    user_query = state["user_query"]
    selected_tables = state.get("selected_tables", [])

    if not selected_tables:
        state["error"] = "SQL Agent: 没有可用的表/字段信息（selected_tables 为空）"
        state["messages"].append(f"[SQL Agent] ERROR: {state['error']}")
        return state

    state["current_step"] = "sql_agent"

    # 构建 schema 描述（V4: 含 dtype + business_term + 表级上下文）
    table_context = state.get("table_context", "")
    schema_info = _build_schema_info_str(selected_tables, table_context=table_context)

    # V4: 日志 — schema 上下文统计
    tables_involved_set = set(item.get("table_name", "") for item in selected_tables)
    has_dtype = any(item.get("dtype") for item in selected_tables)
    has_biz = any(item.get("business_term") for item in selected_tables)
    state["messages"].append(
        f"[SQL Agent] Schema上下文: {len(selected_tables)}字段, "
        f"增强后{len(schema_info)}字符, 涉及{len(tables_involved_set)}张表, "
        f"dtype={'Y' if has_dtype else 'N'}, business_term={'Y' if has_biz else 'N'}"
    )
    state["messages"].append(f"[SQL Agent] 收到 {len(selected_tables)} 个字段，开始生成 SQL...")

    # 尝试生成 + 执行（含自修正闭环）
    max_retries = settings.SQL_MAX_RETRIES
    sql = None
    error_msg = None
    retry_log = []  # 详细重试日志
    task_plan = state.get("task_plan") or {}
    specialist_sql = deterministic_evidence_sql(task_plan)
    query_contract = (
        None
        if specialist_sql or not settings.BUSINESS_SEMANTICS_ENABLED
        else build_query_contract(user_query)
    )
    if not settings.BUSINESS_SEMANTICS_ENABLED:
        state["messages"].append(
            "[SQL Agent] 消融配置：业务语义层已关闭，使用 Schema RAG + LLM"
        )
    contract_sql = specialist_sql or (query_contract.sql if query_contract else None)
    if query_contract:
        task_plan["query_contract"] = query_contract.to_dict()
        state["task_plan"] = task_plan
        state["messages"].append(
            f"[SQL Agent] 命中业务语义配方 {query_contract.recipe_id}: "
            f"{query_contract.rationale}"
        )

    for attempt in range(1, max_retries + 2):  # 1 次初始 + max_retries 次重试
        if attempt == 1:
            if contract_sql:
                sql = contract_sql
                state["messages"].append(
                    "[SQL Agent] 使用专用工具的确定性证据 SQL"
                )
            else:
                # 开放式业务问题仍由 LLM 生成 SQL。
                try:
                    sql = _generate_sql(user_query, schema_info)
                except Exception as e:
                    state["error"] = f"LLM 调用失败: {e}"
                    state["messages"].append(f"[SQL Agent] ERROR: {state['error']}")
                    return state
        else:
            # 自修正
            state["messages"].append(
                f"[SQL Agent] 第{attempt-1}次重试：将错误反馈给 LLM..."
            )
            try:
                sql = _retry_sql(user_query, sql, error_msg, schema_info)
            except Exception as e:
                retry_log.append({
                    "attempt": attempt,
                    "phase": "llm_retry",
                    "error": str(e),
                })
                continue

        # 安全校验
        is_safe, reason = validate_select_only(sql)
        if not is_safe:
            retry_log.append({
                "attempt": attempt,
                "phase": "safety_check",
                "sql": sql,
                "result": "REJECTED",
                "reason": reason,
            })
            error_msg = f"SQL 安全校验失败: {reason}"
            state["messages"].append(f"[SQL Agent] 安全校验不通过: {reason}")
            continue

        # 语义锚点校验：用户明确点名的唯一字段不能被 LLM 自行替换成分箱或代理指标。
        missing_columns = missing_required_columns(sql, selected_tables)
        if missing_columns:
            reason = "生成 SQL 未使用用户明确指定的字段: " + ", ".join(missing_columns)
            retry_log.append({
                "attempt": attempt,
                "phase": "semantic_grounding",
                "sql": sql,
                "result": "REJECTED",
                "reason": reason,
            })
            error_msg = reason
            state["messages"].append(f"[SQL Agent] 语义锚点校验不通过: {reason}")
            continue

        invalid_columns = invalid_qualified_columns(sql)
        if invalid_columns:
            reason = "字段归属校验失败: " + "; ".join(invalid_columns)
            retry_log.append({
                "attempt": attempt,
                "phase": "schema_ownership",
                "sql": sql,
                "result": "REJECTED",
                "reason": reason,
            })
            error_msg = reason
            state["messages"].append(f"[SQL Agent] 字段归属校验不通过: {reason}")
            continue

        convention_issue = percentage_scale_issue(user_query, sql)
        if convention_issue:
            retry_log.append({
                "attempt": attempt,
                "phase": "business_convention",
                "sql": sql,
                "result": "REJECTED",
                "reason": convention_issue,
            })
            error_msg = convention_issue
            state["messages"].append(
                f"[SQL Agent] 业务口径校验不通过: {convention_issue}"
            )
            continue

        # 执行 SQL
        rows, exec_error = _execute_sql(sql)

        log_entry = {
            "attempt": attempt,
            "phase": "execute",
            "sql": sql,
            "result": "SUCCESS" if exec_error is None else "FAILED",
            "error": exec_error,
        }
        retry_log.append(log_entry)

        if exec_error is None:
            if query_contract:
                shape_issue = validate_result_shape(rows, query_contract)
                if shape_issue:
                    retry_log.append({
                        "attempt": attempt,
                        "phase": "result_contract",
                        "sql": sql,
                        "result": "REJECTED",
                        "reason": shape_issue,
                    })
                    error_msg = shape_issue
                    state["messages"].append(
                        f"[SQL Agent] 查询形状契约不通过: {shape_issue}"
                    )
                    continue
            # 成功
            state["sql"] = sql
            state["query_result"] = rows
            state["sql_error"] = None
            state["sql_retries"] = attempt - 1
            state["sql_retry_log"] = retry_log
            state["evidence"] = build_evidence_bundle(
                user_query=user_query,
                task_plan=state.get("task_plan") or {},
                sql=sql,
                rows=rows,
            )
            if query_contract:
                state["evidence"]["query_contract"] = query_contract.to_dict()
            state["messages"].append(
                f"[SQL Agent] SQL 执行成功！返回 {len(rows)} 行数据 "
                f"(尝试 {attempt} 次)"
            )
            # 附加详细重试日志（供评估使用）
            state["messages"].append(
                f"[SQL Agent] 重试日志: {retry_log}"
            )
            return state
        else:
            # 失败，准备重试
            error_msg = exec_error
            state["messages"].append(
                f"[SQL Agent] 第{attempt}次执行失败: {exec_error[:100]}..."
            )

    # 所有尝试都失败
    state["error"] = f"SQL 生成失败（共{max_retries + 1}次尝试）: {error_msg}"
    state["sql"] = sql
    state["sql_error"] = error_msg
    state["sql_retries"] = max_retries + 1
    state["sql_retry_log"] = retry_log
    state["query_result"] = None
    state["messages"].append(f"[SQL Agent] FAILED: {state['error']}")
    state["messages"].append(f"[SQL Agent] 完整重试日志: {retry_log}")

    return state
