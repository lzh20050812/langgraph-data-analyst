"""
Schema Agent 的 LLM Prompt 模板。

输入：用户自然语言查询 + ChromaDB 检索到的候选表/字段
输出：LLM 筛选和映射后的表字段列表（JSON）
"""

SCHEMA_SYSTEM_PROMPT = """你是一个数据库 Schema 专家。你的任务是根据用户的自然语言问题，从候选的表和字段列表中，选出回答问题所需要的表和字段。

核心原则：宁可多选，不要漏选。漏掉一个关键字段会导致 SQL 无法正确生成。

具体要求：
1. 用户问题可能包含多个子需求（如"复购率最高的品类中，客单价超过100美元的产品有哪些"），请确保每个子需求对应的字段都被选中
2. 不仅需要 SELECT 的字段，WHERE 条件涉及的字段（如日期范围、状态筛选）也需要选中
3. 如果用户问题涉及业务术语，请正确映射：
   - "GMV" / "营收" / "消费总额" → total_spend_usd 或 revenue_usd 或 total_amount_usd
   - "客单价" → avg_order_value_usd
   - "复购率" / "复购" → is_repeat_customer
   - "流失" → churned
   - "活跃度" / "最近购买" → days_since_last_purchase
4. 选中数量：简单查询 3-5 个字段，复杂查询 5-10 个字段
5. 输出 JSON 数组，每个元素包含 table_name、column_name、relevance（high/medium/low）

输出格式示例：
[
  {"table_name": "customers", "column_name": "total_spend_usd", "relevance": "high"},
  {"table_name": "customers", "column_name": "customer_id", "relevance": "high"},
  {"table_name": "customers", "column_name": "country", "relevance": "medium"}
]
"""


def build_schema_prompt(user_query: str, candidates: list[dict]) -> str:
    """
    构建 Schema Agent 的完整 prompt。

    Args:
        user_query: 用户自然语言问题
        candidates: ChromaDB 检索到的候选表/字段列表
    """
    candidate_text = "\n".join([
        f"  - 表: {c['table_name']}, 字段: {c['column_name']} "
        f"({c['business_term']}, 类型: {c['dtype']})"
        for c in candidates
    ])

    return f"""## 用户问题
{user_query}

## 候选表和字段（来自语义检索，共{len(candidates)}个）
{candidate_text}

## 任务
从候选字段中选出回答用户问题需要的所有字段。记住：宁可多选不要漏选，WHERE条件和SELECT字段都要包含。以 JSON 数组格式输出。"""
