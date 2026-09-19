"""
Report Agent 的 LLM Prompt 模板。

输入：用户查询 + Analysis Agent 结果 + Prediction Agent 结果 + 数据治理评分
输出：结构化的经营洞察报告（含"为什么"和"怎么办"，不是罗列数字）
"""

REPORT_SYSTEM_PROMPT = """你是一位资深的企业经营分析顾问。你的任务是根据提供的数据分析结果和预测结果，撰写一份简洁精准的经营洞察报告。

## 输出约束（必须遵守）
- 总字数：严格控制在 800-1500 字以内
- 直接输出报告正文，不要前言、不要结尾总结、不要客套话
- 用简洁的数据陈述代替长篇推理
- 只写最重要的发现，不做面面俱到的流水账

## 核心原则
- **证据约束**：事实和数字只能来自基础查询证据、数据分析结果、预测结果和数据质量评估，不得自行补造
- **事实引用**：每个数值结论必须在同一行引用基础查询事实编号 `[F#]`；没有对应事实编号的数字不得写入报告
- **解释数据**：每个数据点都要解释"这意味着什么"和"应该怎么应对"
- **聚焦关键**：只写最关键的 2-4 条发现，不是越多越好
- **可执行**：建议必须具体、可量化、可落地
- **诚实标注**：数据质量有问题的结论要明确标注可信度

## 报告结构（使用 Markdown）

### 一、数据摘要
用 3-5 句话概述本次分析覆盖的数据范围、核心指标和整体评价。列出最关键的 3-5 个数字。

### 二、关键发现 (2-4 条)
每条格式：
- **发现标题**：（一句话概括核心洞察）
  - 数据支撑：（引用具体数字）
  - 商业解读：（为什么重要，意味着什么）

### 三、根因分析
对关键问题进行归因分析，解释背后的业务驱动因素。如果有客户分层数据，说明各层客户的核心差异和导致差异的可能原因。

### 四、行动建议 (2-3 条)
每条格式：
1. **【高/中/低优先级】建议**
   - 依据：（引用数据）
   - 措施：（具体可执行动作）
   - 预期效果：（仅在证据中存在对应事实时量化，否则写“需进一步实验评估”）

## 风格
- 专业但不晦涩，面向管理层
- 用数字支撑论点，但不堆砌数字
- 每条发现控制在 80-150 字"""


def build_report_prompt(
    user_query: str,
    analysis_result: dict,
    prediction_result: dict,
    governance_result: dict = None,
    evidence: dict = None,
    historical_memory: list = None,
) -> str:
    """
    构建 Report Agent 的完整 prompt。

    Args:
        user_query: 用户原始自然语言问题
        analysis_result: Analysis Agent 的输出（RFM + K-Means + 运营指标）
        prediction_result: Prediction Agent 的输出（流失预测 + 销售预测）
        governance_result: 数据治理Agent 的质量评分（可选）
        evidence: SQL、结果行和节点产出的可追溯证据包（可选）
    """
    import json

    # 格式化分析结果
    analysis_text = _format_analysis_for_prompt(analysis_result)

    # 格式化预测结果
    prediction_text = _format_prediction_for_prompt(prediction_result)

    # 格式化数据治理结果
    gov_text = ""
    if governance_result:
        gov_text = _format_governance_for_prompt(governance_result)

    evidence_text = _format_evidence_for_prompt(evidence or {})
    memory_text = _format_memory_for_prompt(historical_memory or [])

    return f"""## 用户的问题
{user_query}

## 基础查询证据
{evidence_text}

## 数据分析结果
{analysis_text}

## 预测结果
{prediction_text}

## 数据质量评估
{gov_text if gov_text else "（未进行数据质量评估）"}

## 历史分析经验（只能复用方法和策略框架，不得把历史数字当作本次事实）
{memory_text}

## 任务
请根据以上信息，撰写一份简洁的经营洞察报告。关键要求：
1. 严格控制在 800-1500 字，只写最重要的发现
2. 每个关键数据解释"为什么"重要，并给出"怎么办"
3. 给出 2-3 条可执行的具体策略建议
4. 标注数据可信度（如有数据质量问题）
5. 以上四类输入是唯一允许引用的事实来源；若证据不足，明确写“当前证据不足”，不得补造数字或原因
6. 使用 Markdown 格式，保持简洁，不要前言和结尾总结"""


def _format_memory_for_prompt(memories: list) -> str:
    if not memories:
        return "（无可复用历史分析）"
    parts = []
    for memory in memories[:3]:
        parts.append(
            f"- 相似问题：{memory.get('query', '')}\n"
            f"  相似度：{memory.get('score', 0)}\n"
            f"  历史内容：{str(memory.get('document', ''))[:1200]}"
        )
    return "\n".join(parts)


def _format_evidence_for_prompt(evidence: dict) -> str:
    """格式化有界证据视图，避免把整个查询结果塞入 prompt。"""
    if not evidence:
        return "（无可追溯证据；不得生成具体数字）"

    import json

    rows = evidence.get("rows") or []
    payload = {
        "sql": evidence.get("sql"),
        "row_count": evidence.get("row_count", len(rows)),
        "columns": evidence.get("columns") or [],
        "rows_sample": rows[:30],
        "task_plan": evidence.get("task_plan") or {},
        "metric_catalog_version": evidence.get("metric_catalog_version"),
        "source_tables": evidence.get("source_tables") or [],
        "facts": evidence.get("facts") or [],
        "validation": evidence.get("validation") or {},
        "limitations": evidence.get("limitations") or [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _format_analysis_for_prompt(analysis: dict) -> str:
    """将 Analysis Agent 的输出格式化为 prompt 可读文本。"""
    if not analysis:
        return "（无分析数据）"

    parts = []

    query_analysis = analysis.get("query_analysis", {})
    if query_analysis:
        parts.append("### SQL 查询结果摘要")
        parts.append(f"  - 返回行数: {query_analysis.get('row_count', 0)}")
        columns = query_analysis.get("columns", [])
        if columns:
            parts.append(f"  - 字段: {', '.join(columns)}")
        for column, summary in query_analysis.get("numeric_summary", {}).items():
            parts.append(
                f"  - {column}: 均值={summary.get('mean')}, "
                f"最小值={summary.get('min')}, 最大值={summary.get('max')}"
            )

    # RFM 分段
    rfm = analysis.get("rfm", {})
    segments = rfm.get("segments", {})
    if segments:
        total = sum(segments.values())
        parts.append("### RFM 客户价值分段")
        for seg, count in sorted(segments.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100 if total > 0 else 0
            parts.append(f"  - {seg}: {count} 人 ({pct:.1f}%)")

    # 运营指标
    metrics = analysis.get("metrics", {})
    if metrics:
        parts.append("\n### 核心运营指标")
        if "gmv_usd" in metrics:
            gmv = float(metrics['gmv_usd']) if metrics.get('gmv_usd') is not None else 0
            parts.append(f"  - GMV: ${gmv:,.0f}")
        if "avg_order_value_usd" in metrics:
            aov = float(metrics['avg_order_value_usd']) if metrics.get('avg_order_value_usd') is not None else 0
            parts.append(f"  - 平均客单价: ${aov:,.2f}")
        if "repeat_purchase_rate" in metrics:
            rpr = float(metrics['repeat_purchase_rate']) if metrics.get('repeat_purchase_rate') is not None else 0
            parts.append(f"  - 复购率: {rpr*100:.1f}%")
        if "churn_rate" in metrics:
            cr = float(metrics['churn_rate']) if metrics.get('churn_rate') is not None else 0
            parts.append(f"  - 流失率: {cr*100:.1f}%")

    # 品类分布
    cat_rev = metrics.get("category_revenue", {})
    if cat_rev:
        parts.append("\n### 品类收入分布")
        for cat, rev in sorted(cat_rev.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"  - {cat}: ${rev:,.0f}")

    # 会员分布
    member = metrics.get("membership_distribution", {})
    if member:
        parts.append("\n### 会员等级分布")
        for tier, count in member.items():
            parts.append(f"  - {tier}: {count} 人")

    # K-Means
    kmeans = analysis.get("kmeans", {})
    if kmeans and kmeans.get("silhouette_score"):
        parts.append(f"\n### K-Means 聚类 (轮廓系数: {kmeans['silhouette_score']})")
        parts.append(f"  - 聚类数: {kmeans.get('n_clusters', 'N/A')}")
        parts.append(f"  - 样本量: {kmeans.get('sample_count', 'N/A')}")
        sizes = kmeans.get("cluster_sizes", {})
        for cid, size in sizes.items():
            parts.append(f"  - 簇 {cid}: {size} 人")

    return "\n".join(parts)


def _format_prediction_for_prompt(prediction: dict) -> str:
    """将 Prediction Agent 的输出格式化为 prompt 可读文本。"""
    if not prediction:
        return "（无预测数据）"

    parts = []

    # 流失预测
    churn = prediction.get("churn", {})
    if churn and "auc" in churn:
        churn_model = churn.get("model") or churn.get("selected_model") or "未标注模型"
        parts.append(f"### {churn_model} 流失预测模型")
        parts.append(f"  - AUC: {churn['auc']}")
        parts.append(f"  - 正样本率（流失客户占比）: {churn.get('positive_rate', 0)*100:.1f}%")

        optimal = churn.get("optimal_threshold", {})
        if optimal:
            parts.append(f"  - 最优阈值: {optimal.get('threshold', 'N/A')}")
            parts.append(f"  - Recall@最优: {optimal.get('recall', 0)*100:.1f}%")
            parts.append(f"  - Precision@最优: {optimal.get('precision', 0)*100:.1f}%")
            parts.append(f"  - F1@最优: {optimal.get('f1', 'N/A')}")

        # Top 特征
        top_features = churn.get("feature_importance", [])[:5]
        if top_features:
            parts.append("  - 流失关键因子（Top 5）:")
            for f in top_features:
                imp = float(f['importance']) if f.get('importance') is not None else 0
                parts.append(f"    · {f['feature']}: importance={imp:.4f}")

        # 高风险客户
        top_risk = churn.get("top_risk_customers", [])[:5]
        if top_risk:
            parts.append("  - 流失风险最高的5位客户:")
            for c in top_risk:
                prob = float(c['churn_probability']) if c.get('churn_probability') is not None else 0
                parts.append(f"    · customer_id={c['customer_id']}, 流失概率={prob:.2%}")

    # 销售预测
    sales = prediction.get("sales", {})
    if sales and "rmse" in sales:
        rmse = float(sales['rmse']) if sales.get('rmse') is not None else 0
        mae = float(sales['mae']) if sales.get('mae') is not None else 0
        sales_model = sales.get("model") or sales.get("selected_model") or "未标注模型"
        parts.append(f"\n### {sales_model} 销售预测模型")
        parts.append(f"  - RMSE: {rmse:,.0f}")
        parts.append(f"  - MAPE: {sales.get('mape_pct', 'N/A')}%")
        parts.append(f"  - MAE: {mae:,.0f}")
        parts.append(f"  - 训练期: {sales.get('train_periods', 'N/A')} 个月")
        parts.append(f"  - 测试期: {sales.get('test_periods', 'N/A')} 个月")

        # 未来预测摘要
        forecast = sales.get("forecast", [])
        if forecast:
            future_only = [f for f in forecast if f.get("phase") == "future"]
            backtest_only = [f for f in forecast if f.get("phase") == "backtest"]
            legacy_rows = [f for f in forecast if not f.get("phase")]
            if backtest_only:
                parts.append(f"  - 回测期: {len(backtest_only)} 个周期（不计入未来预测）")
            if legacy_rows:
                parts.append("  - 兼容提示: 部分旧结果缺少 phase，未当作未来预测")
            if future_only:
                parts.append("  - 未来预测摘要:")
                for f in future_only[:6]:
                    yhat = float(f.get('yhat', 0)) if f.get('yhat') is not None else 0
                    lo, hi = f.get('yhat_lower'), f.get('yhat_upper')
                    interval = (
                        f"(区间: ${float(lo):,.0f} ~ ${float(hi):,.0f})"
                        if lo is not None and hi is not None and float(lo) != float(hi)
                        else "(未提供有效预测区间)"
                    )
                    parts.append(f"    · {f['ds']}: 预测=${yhat:,.0f} {interval}")
            else:
                parts.append("  - 未来预测: 无标记为 future 的结果")

    return "\n".join(parts)


def _format_governance_for_prompt(gov: dict) -> str:
    """将数据治理结果格式化为 prompt 可读文本。"""
    if not gov:
        return "（无数据质量评估）"

    parts = []
    score = gov.get("quality_score", "N/A")
    parts.append(f"### 数据质量评分: {score}")

    warnings = gov.get("warnings", [])
    if warnings:
        parts.append("### 数据质量问题")
        for w in warnings:
            parts.append(f"  - [!] {w}")

    columns_detail = gov.get("columns", {})
    if columns_detail:
        parts.append("### 字段级质量")
        for col, detail in columns_detail.items():
            if detail.get("null_rate", 0) > 0.1:
                parts.append(f"  - {col}: 缺失率 {detail['null_rate']*100:.1f}%")

    return "\n".join(parts)
