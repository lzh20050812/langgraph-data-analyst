"""
SQL Agent 的 LLM Prompt 模板。

输入：用户问题 + Schema Agent 选中的表/字段信息
输出：MySQL 可执行的 SQL 语句
"""

SQL_SYSTEM_PROMPT = """你是一个 SQL 专家。你的任务是根据用户的自然语言问题和可用的数据库表结构，生成可执行的 MySQL SELECT 语句。

## 数据库信息
数据库名: bi_she
数据库类型: MySQL 8.0

## 规则
1. 只生成 SELECT 语句，不要生成 INSERT/UPDATE/DELETE/DROP 等修改数据的语句
2. 只使用提供的表和字段，不要编造不存在的表名或字段名
3. 使用标准的 MySQL 语法
4. 聚合查询记得使用 GROUP BY
5. 排序使用 ORDER BY，默认降序排列
6. 限制返回行数时使用 LIMIT
7. 如果用户问题涉及日期范围，使用 DATE 类型字段进行比较（如 WHERE order_date >= '2025-01-01'）
8. 只返回纯 SQL 语句，不要包含解释文字，不要用 ```sql 包裹
"""

SQL_RETRY_PROMPT = """之前的 SQL 执行失败了。请根据错误信息修正 SQL。

## 原始问题
{user_query}

## 上次生成的 SQL
{sql}

## 执行错误信息
{error}

## 要求
请分析错误原因并生成修正后的 SQL。只返回纯 SQL 语句，不要包含其他文字。"""


def build_sql_prompt(user_query: str, schema_info: str) -> str:
    """构建 SQL 生成的 prompt。"""
    return f"""## 可用的表和字段
{schema_info}

## 用户问题
{user_query}

## 任务
请根据以上信息，生成一条 MySQL SELECT 语句来回答用户的问题。"""


def build_sql_retry_prompt(user_query: str, sql: str, error: str) -> str:
    """构建 SQL 修正的 prompt。"""
    return SQL_RETRY_PROMPT.format(user_query=user_query, sql=sql, error=error)
