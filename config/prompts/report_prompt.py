"""
Report Agent 的 LLM Prompt 模板。

输入：用户查询 + Analysis Agent 结果 + Prediction Agent 结果 + 数据治理评分
输出：结构化的经营洞察报告（含"为什么"和"怎么办"，不是罗列数字）
"""

REPORT_SYSTEM_PROMPT = """你是一位资深的企业经营分析顾问。你的任务是根据提供的数据分析结果和预测结果，撰写一份结构化的经营洞察报告。

## 报告要求

### 1. 核心原则
- **不只是罗列数字**：每个数据点都要解释"这意味着什么"（why）和"应该怎么做"（how）
- **优先级排序**：把最重要的发现放在前面，不要面面俱到地报流水账
- **可执行建议**：每条策略建议都应该是具体的、可落地的，而不是泛泛而谈的"加强管理""优化体验"
- **诚实面对数据**：如果数据质量有问题（如缺失率高），在报告中如实标注该结论的可信度

### 2. 报告结构
请按以下结构组织报告（使用 Markdown 格式）：

## 一、核心发现摘要
- 3-5 条最重要的发现，每条 1-2 句话
- 标注每条发现的可信度（高/中/低，基于数据质量）

## 二、客户价值分析
- RFM 分段解读：各价值段客户的占比和特征
- K-Means 聚类解读：各客户群体的特征画像和商业含义
- 关键洞察和针对性策略

## 三、运营绩效诊断
- GMV / 客单价 / 复购率 / 流失率的核心数据解读
- 品类收入分布的健康度判断
- 会员等级分布的合理性分析
- 月度营收趋势的异常点和拐点解读

## 四、预测与预警
- 客户流失风险：高风险客户画像、预测可信度、建议的挽留策略
- 销售趋势预测：未来6个月的趋势判断、季节性波动提示、异常预警
- 预测模型的局限性说明（基于实际评估指标）

## 五、策略建议（按优先级排序）
每条建议格式：
1. **【高/中/低优先级】建议标题**
   - 数据依据：（引用具体数字）
   - 具体措施：（2-3 条可执行的动作）
   - 预期效果：（量化的预期改善幅度，如"预计可提升复购率 2-5pp"）

## 六、数据质量说明
- 标注本次分析中数据质量的潜在问题
- 对可信度较低的结论进行说明

### 3. 风格要求
- 使用专业但不晦涩的语言，目标读者是企业管理层
- 适当使用数字来支撑论点，但不要堆砌数字
- 每条发现控制在 50-150 字
- 总报告控制在 1500-3000 字
"""


def build_report_prompt(
    user_query: str,
    analysis_result: dict,
    prediction_result: dict,
    governance_result: dict = None,
) -> str:
    """
    构建 Report Agent 的完整 prompt。

    Args:
        user_query: 用户原始自然语言问题
        analysis_result: Analysis Agent 的输出（RFM + K-Means + 运营指标）
        prediction_result: Prediction Agent 的输出（流失预测 + 销售预测）
        governance_result: 数据治理Agent 的质量评分（可选）
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

    return f"""## 用户的问题
{user_query}

## 数据分析结果
{analysis_text}

## 预测结果
{prediction_text}

## 数据质量评估
{gov_text if gov_text else "（未进行数据质量评估）"}

## 任务
请根据以上信息，撰写一份结构化的经营洞察报告。记住：
1. 解释每个关键数据"为什么"重要
2. 给出"怎么办"的具体策略建议
3. 标注数据可信度（如果数据质量有问题）
4. 使用 Markdown 格式输出完整报告"""


def _format_analysis_for_prompt(analysis: dict) -> str:
    """将 Analysis Agent 的输出格式化为 prompt 可读文本。"""
    if not analysis:
        return "（无分析数据）"

    parts = []

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
        parts.append("### XGBoost 流失预测模型")
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
        parts.append("\n### Prophet 销售预测模型")
        parts.append(f"  - RMSE: {rmse:,.0f}")
        parts.append(f"  - MAPE: {sales.get('mape_pct', 'N/A')}%")
        parts.append(f"  - MAE: {mae:,.0f}")
        parts.append(f"  - 训练期: {sales.get('train_periods', 'N/A')} 个月")
        parts.append(f"  - 测试期: {sales.get('test_periods', 'N/A')} 个月")

        # 未来预测摘要
        forecast = sales.get("forecast", [])
        if forecast:
            future_only = [f for f in forecast if f.get("ds", "") > "2026"]
            if future_only:
                parts.append("  - 未来预测摘要:")
                for f in future_only[:6]:
                    yhat = float(f.get('yhat', 0)) if f.get('yhat') is not None else 0
                    lo = float(f.get('yhat_lower', 0)) if f.get('yhat_lower') is not None else 0
                    hi = float(f.get('yhat_upper', 0)) if f.get('yhat_upper') is not None else 0
                    parts.append(
                        f"    · {f['ds']}: "
                        f"预测=${yhat:,.0f} "
                        f"(区间: ${lo:,.0f} ~ ${hi:,.0f})"
                    )

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
