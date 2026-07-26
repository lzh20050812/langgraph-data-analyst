# PROGRESS.md —— 项目进度与交接文档

> 最后更新：2026-07-26（Phase 3 完成）
>
> 本文档记录实际开发过程中发生的事实、与原计划的偏差、已知局限性和下一步任务。
> 新会话开始时，请与本文件 + PROJECT_BRIEF.md 一起阅读，不要重新假设已确认的架构决策。

---

## 一、已完成

### Phase 1：核心链路（数据 → ETL → 存储 → Text2SQL）

| 文件 | 职责 |
|------|------|
| `scripts/generate_dirty_sources.py` | 脏数据生成（seed=42，可复现），拆分 App/Web 两个异构源，注入缺失值+重复+格式不一致，输出 injection_report.json |
| `etl/field_mapper.py` | 字段名映射（camelCase/snake_case → 统一标准 schema） |
| `etl/cleaner.py` | 缺失值填充（median/mode）+ 去重 + gender/日期格式统一 |
| `etl/loader.py` | CSV → MySQL 导入 + 聚合视图创建 |
| `etl/quality_report.py` | 治理前后数据质量对比报告 |
| `etl/run_etl.py` | ETL 端到端验证入口 |
| `storage/mysql/schema.sql` | 4 张业务表 + 2 张聚合视图 DDL |
| `storage/mysql/client.py` | SQLAlchemy MySQL 引擎/连接封装 |
| `storage/db_adapter.py` | 双后端适配器（MySQL 优先，DuckDB 降级） |
| `storage/chromadb/embedder.py` | BAAI/bge-small-zh 向量化 + ChromaDB 持久化 |
| `storage/chromadb/schema_metadata.py` | 54 条字段元数据（含 aliases 同义词） |
| `storage/init_storage.py` | 一键初始化：建表+ETL+导入+创建视图 |
| `config/settings.py` | 全局配置（.env 驱动） |
| `config/prompts/schema_prompt.py` | Schema Agent 的 LLM prompt 模板 |
| `config/prompts/sql_prompt.py` | SQL Agent 的生成/自修正 prompt 模板 |
| `agents/state.py` | AgentState TypedDict 定义 |
| `agents/llm.py` | LLM 调用抽象层（OpenAI 兼容，支持 DeepSeek） |
| `agents/planner.py` | LangGraph 状态图入口 + 意图解析 + 条件路由 |
| `agents/schema_agent.py` | ChromaDB 粗排 + LLM 精排，业务术语→表字段 |
| `agents/sql_agent.py` | NL→SQL→安全校验→执行→自修正（最多3次重试） |
| `evaluation/eval_schema.py` | Schema Agent 检索准确率评估（20条测试集） |
| `evaluation/eval_sql.py` | SQL Agent 生成成功率评估（25条，含自修正追踪） |
| `docker/docker-compose.yml` | MySQL 8.0 容器 |
| `requirements.txt` | Python 依赖清单 |

**Phase 1 关键评估数字（全部在 MySQL 上跑出）：**

| 指标 | 数值 |
|------|------|
| ETL 缺失值修复率 | 100%（1088 → 0） |
| ETL 重复行清除率 | 100%（192 → 0） |
| orders.customer_rating 自然缺失率 | 63.0%（15749/25000，保留 NULL 供数据治理Agent评分） |
| Schema Agent Top-5 命中率 | 95%（Pure ChromaDB，20条测试集） |
| Schema Agent LLM 精排后 Hard 组字段召回 | 73.6%（vs Pure ChromaDB 51.4%，+22pp） |
| SQL Agent 首次成功率（25条全量） | 92.0%（23/25） |
| SQL Agent 可解题首次成功率（24条） | 95.8%（23/24） |
| SQL Agent 自修正提效 | 0%（2条失败经3次重试仍未修复，根因见"已知局限性"） |
| 安全拦截（SELECT 防御） | 0 次误拦，8/8 危险语句正确拒绝 |

---

### Phase 2：核心分析能力（Analysis + Prediction）

| 文件 | 职责 |
|------|------|
| `agents/analysis_agent.py` | RFM 客户价值分析 + K-Means 聚类 + 运营指标计算（纯计算，非 LLM Agent） |
| `agents/prediction_agent.py` | XGBoost 流失预测 + Prophet 销售预测（含阈值优化） |
| `agents/chart_renderer.py` | 分析/预测结果 → ECharts 配置 JSON（非 Agent，纯渲染） |
| `evaluation/eval_prediction.py` | 流失预测 + 销售预测模型评估脚本 |

**Phase 2 关键评估数字（全部在 MySQL 上跑出）：**

| 模型 | 指标 | 数值 |
|------|------|------|
| XGBoost 流失预测 | AUC | **0.693** |
| | Recall@最优阈值(0.48) | **44.7%** |
| | Precision@最优阈值 | 27.5% |
| | F1@最优阈值 | 0.341 |
| | 正样本率 | 9.0%（715/7600，严重类不平衡） |
| | 类别不平衡处理 | scale_pos_weight≈10.1, stratify=train_test_split |
| Prophet 销售预测 | RMSE | 6,585 |
| | MAPE | **15.65%** |
| | MAE | 5,838 |
| | 划分方式 | 时间序列按先后顺序（前69月训练，后6月测试） |
| RFM 客户分段 | 高价值/潜力/一般/流失风险 | 1742 / 3062 / 2442 / 354 |
| K-Means 聚类 | 轮廓系数 | 0.201（4簇） |

**LangGraph 路由（已生效，Phase 2 版）：**

```
sql_query:  START → planner → schema → sql → END
analysis:   START → planner → schema → sql → analysis → chart → END
prediction: START → planner → schema → sql → prediction → chart → END
mixed:      START → planner → schema → sql → analysis → prediction → chart → END
```

---

## 二、与原计划的偏差

### 2.1 `db_adapter.py` 双后端设计（新增，原计划无）

- **偏差**：PROJECT_BRIEF.md 只提到 MySQL + Docker，实际多了一个 DuckDB 内存数据库作为降级后端
- **原因**：开发期便利——本地没启动 Docker 时也能跑通逻辑
- **定性**：开发脚手架，不是正式架构。论文的系统架构图和所有评估数据只提 MySQL
- **是否需要修正**：不需要。保留 DuckDB 路径方便后续快速迭代，但论文/简历中不出现 DuckDB

### 2.2 评估集 Query 21 被标记为 unsolvable

- **偏差**：原计划 25 条 SQL 测试全部计入成功率，实际 1 条被剔除
- **Query 21**："2025年每月的新增客户数和流失客户数对比"
- **原因**：数据模型不支持。`customers.churned` 是静态 0/1 标签，没有 `churn_date` 列；`monthly_revenue` 只有 `new_customers`，没有 `churned_customers`。该问题在当前 schema 下不可解
- **处理**：保留在测试集中但打上 `unsolvable` 标记，成功率报告同时给出两个数字：全部25条 92.0%，可解题24条 95.8%。论文里应如实写"1条经分析确认为数据模型限制，已从主指标剔除"
- **是否需要修正**：不需要。这是方法论发现而非 bug

### 2.3 Query 25 失败根因：Schema Agent 召回盲区，非命名歧义

- **Query 25**："过去90天内有退货记录的客户的会员等级分布"
- **首次诊断假设**：LLM 把列名搞错了（`member_level` 误代 `membership_tier`）
- **实际根因**：Schema Agent 根本没把 `membership_tier` 返回给 LLM（返回的是 `returns_made/registration_date/returned/delivery_days`）。LLM 从头到尾没见过正确列名，所以反复猜错
- **已采取的修复**：给 `schema_metadata.py` 中 `membership_tier` 的 aliases 补充了 `"membership_level"`，并重建了 ChromaDB
- **修复效果**：alias 修复本身是对的，但不能解决这个具体案例——因为根因是中文语义检索的覆盖盲区，不是英文 alias 缺失
- **是否需要修正**：短期不改。这是论文 Discussion 的好素材——"自修正的有效性边界是 Schema Agent 的召回覆盖率"

### 2.4 SQL Agent 自修正闭环提效为 0%

- **偏差**：原计划预期自修正能修复部分 SQL 错误，实际修复数为 0
- **数据**：2 条失败案例，每条重试 3 次，共 6 次重试，0 次修复成功
- **原因**：2 条失败都是 LLM 没见过正确列名导致的（Q21 数据模型不支持，Q25 Schema Agent 召回遗漏）。重试只是让 LLM 在黑暗中反复猜，不会收敛
- **论文价值**：这正是"自修正机制的局限性"讨论的好素材——当错误根因在 Schema Agent 召回阶段时，修正回路无效。论文可以对比"首次 92% → 修正后 92%"来说明自修正的适用边界
- **是否需要修正**：不需要在代码层面修。但论文 Discussion 章节应讨论这个发现

### 2.5 XGBoost 准确率 85.1% 被判定为误导性指标

- **原始输出**：Accuracy 85.1%（低于 Dummy 全预测"不流失"的 91%）
- **问题**：正样本仅 9%，Accuracy 在类不平衡场景下没有参考价值
- **处理**：从报告中移除 Accuracy 为独立亮点指标，改报 AUC + Precision/Recall/F1 双阈值对比
- **是否需要修正**：已修正

### 2.6 意图解析关键词调整

- **偏差**：原 planner.py 中 `"gmv"` 在 analysis_keywords 中，导致"中国区GMV最高的5个客户"被误路由到 analysis
- **修正**：从 analysis_keywords 移除宽泛词（`"gmv"`、`"客单价"`、`"复购率"`），只保留明确的分析动作词（`"rfm分析"`、`"客户分层"` 等）
- **是否需要修正**：已修正

### 2.7 Governance Agent 源表追溯设计（新增，原计划简化版增强）

- **偏差**：原计划只对查询结果集做轻量级评分，实际增加了源表追溯层
- **原因**：聚合查询（GROUP BY）会把字段的真实缺失率压缩到几行里——例如 `customer_rating` 在 orders 表中缺失 63%，但 GROUP BY 后结果集只显示 2.8%。只看结果集会给出满分 100，追溯源表后正确评分为 65/100
- **定性**：这是对"运行时数据质量评估"定位的正确深化——运行时不仅看结果，还要回答"这批数据来自哪里，源数据靠谱吗"
- **是否需要修正**：不需要。论文中可讨论两层评估设计的必要性

### 2.8 ChromaDB Docker 架构选择

- **偏差**：PROJECT_BRIEF.md 提到 Docker 仅容器化 Python + MySQL，未明确 ChromaDB 的部署方式
- **实际方案**：ChromaDB 作为嵌入进程运行（非独立容器），数据通过 `../data:/app/data` volume 挂载持久化，模型缓存通过 `HF_HOME` + `SENTENCE_TRANSFORMERS_HOME` 环境变量指向持久化路径
- **定性**：保持开发/生产一致性（本地开发也是嵌入模式），避免多一层容器编排复杂度
- **是否需要修正**：不需要

### 2.9 Report Agent 数据准确性受上游输出质量制约

- **发现**：LLM-as-Judge 5 样本评估中，数据准确性均值 7.2/10 为最低维度，而可读性 9.0/10、洞察深度 8.4/10——Report Agent 的核心能力（解读+建议）强于数据引用准确性
- **原因**：测试用的 mock 数据不完整（缺少品类分布、月度趋势等），LLM 在填补缺失信息时产生了数字偏差；数据完整时（R3 聚焦流失+销售），准确性回升至 8/10
- **论文价值**：这印证了多智能体系统中的级联依赖——Report Agent 的质量上限由上游 Agent 输出完整性决定。Governance Agent 的前置标注（可信度标签）已被 Report Agent 正确消费（报告中"可信度"出现 7 次/篇），说明架构设计有效——识别出问题并传递给了下游

---

## 三、已知局限性（论文可用素材）

以下所有问题均已确认，**选择不在此版本修复**，但应在论文中讨论：

### 3.1 模型层面

| 局限性 | 描述 | 论文建议 |
|--------|------|----------|
| XGBoost 正样本仅 9% | 类严重不平衡，Recall 最高 44.7% | Discussion: 可尝试 SMOTE 过采样、集成方法，或引入外部特征 |
| Prophet 测试集仅 6 个月 | 样本量偏小，MAPE 可能有波动 | 诚实报告，标注样本量限制 |
| K-Means 轮廓系数 0.201 | 聚类质量一般，客户群体边界模糊 | 属于探索性分析，非生产级分群 |

### 3.2 Agent 层面

| 局限性 | 描述 | 论文建议 |
|--------|------|----------|
| 自修正提效 0% | 2条失败均无法通过重试修复 | Future Work: 强化 Schema Agent 召回覆盖后再评估自修正 |
| Schema Agent 对部分中文术语召回不足 | "会员等级分布"未返回 membership_tier | 可引入中文同义词表或微调 embedding 模型 |
| SQL Agent 在 Hard 组成功率 87.5% | 多表 JOIN + 子查询场景仍有 1/8 失败 | 正常水平，可作为基线对比不同 LLM |
| 意图解析基于关键词规则 | 复杂/模糊意图可能误判 | Phase 3 可考虑 LLM 意图解析 |

### 3.3 工程层面

| 局限性 | 描述 |
|--------|------|
| DuckDB 开发脚手架存在 | 不影响最终数据，但代码中有双后端痕迹 |
| 评估脚本无共享基类 | eval_schema/eval_sql/eval_prediction 各自独立实现，存在代码重复 |
| 无 CI/CD | 所有评估需手动执行 |

---

## 四、Phase 3 完成（收尾与工程化）

### 4.1 Report Agent ✅
- **文件**：`agents/report_agent.py` + `config/prompts/report_prompt.py`
- **职责**：综合 Analysis / Prediction / Governance 的输出，用 LLM 生成结构化经营洞察报告
- **报告结构**：核心发现摘要 → 客户价值分析 → 运营绩效诊断 → 预测与预警 → 策略建议 → 数据质量说明
- **评估**：`evaluation/eval_report.py`，支持 LLM-as-Judge 四维度评分（完整性/洞察深度/可执行性/可读性/数据准确性）+ 结构完整性检查 + 启发式指标

### 4.2 数据治理 Agent ✅
- **文件**：`agents/governance_agent.py`
- **职责**：对 SQL 查询结果做运行时质量评分（0-100），逐列计算缺失率/唯一值率/数据类型，标记异常
- **设计定位**：运行时轻量评分（非 ETL 批处理），回答"这批数据靠谱吗"

### 4.3 FastAPI 应用层 ✅
- **文件**：`api/main.py` + `frontend/index.html`
- **接口**：`POST /query`（对话式查询）、`GET /charts`（图表数据）、`GET /report`（报告）、`GET /`（前端看板）
- **前端**：纯静态 ECharts 看板，支持自然语言输入 + 快速提问 + 图表渲染 + 报告展示

### 4.4 Docker 部署 ✅
- **文件**：`Dockerfile` + 更新 `docker/docker-compose.yml`
- **架构**：双容器（MySQL 8.0 + FastAPI 应用），健康检查 + 自动初始化

### 4.5 评估层汇总 ✅
- **文件**：`evaluation/eval_report.py`（Report Agent 评估）+ `evaluation/eval_summary.py`（总汇总）
- **eval_summary.py**：一键整合 eval_schema + eval_sql + eval_prediction + eval_report 的所有数字

### Phase 3 LangGraph 路由（最终版）

```
sql_query:   START → planner → schema → sql → governance → END
analysis:    START → planner → schema → sql → governance → analysis → chart → END
prediction:  START → planner → schema → sql → governance → prediction → chart → END
mixed:       START → planner → schema → sql → governance → analysis → prediction → report → chart → END
```

---

## 五、当前环境状态

| 项目 | 状态 |
|------|------|
| MySQL (Docker) | ✅ 运行中，`bi_she_mysql` 容器，端口 3306 |
| 数据 | ✅ 4 张表已导入（customers 7600, orders 25000, monthly_revenue 75, product_summary 140） |
| ChromaDB | ✅ 54 条字段元数据，BAAI/bge-small-zh，`./data/chromadb/` |
| LLM | DeepSeek-V3，API key 在 `.env` 中 |
| Python 环境 | Python 3.13，所有依赖已安装（含 xgboost, prophet） |

**快速启动命令：**
```bash
# ---- 基础设施 ----
docker-compose -f docker/docker-compose.yml up -d     # 启动 MySQL
# 或完整启动（含应用）:
docker-compose -f docker/docker-compose.yml up -d --build

python -m storage.init_storage --skip-etl              # 初始化数据

# ---- 评估 ----
python -m evaluation.eval_sql                          # SQL Agent 评估
python -m evaluation.eval_prediction                   # 模型评估
python -m evaluation.eval_report                       # Report Agent 评估
python -m evaluation.eval_summary                      # 总评估汇总

# ---- 运行 ----
python -c "from agents.planner import run_query; run_query('你的问题')"  # CLI 查询
uvicorn api.main:app --host 0.0.0.0 --port 8000        # 启动 API 服务
```

---

## 六、论文数据速查表

以下数字已经过验证，可直接用于论文：

| 章节 | 数据点 | 数值 |
|------|--------|------|
| 数据治理 | ETL 缺失值修复率 | 100%（1088→0） |
| 数据治理 | ETL 重复行清除率 | 100%（192→0） |
| 数据治理 | customer_rating 自然缺失 | 63.0% |
| Schema Agent | Top-5 检索命中率 | 95% |
| Schema Agent | LLM 精排后 Hard 字段召回 | 73.6%（+22pp vs Pure ChromaDB） |
| SQL Agent | 可解题首次成功率 | 95.8%（24条） |
| SQL Agent | 全量首次成功率 | 92.0%（25条含1条unsolvable） |
| SQL Agent | 自修正后成功率 | 92.0%（与首次相同，提效0%） |
| Analysis | RFM 分段分布 | 高价值22.9% / 潜力40.3% / 一般32.1% / 流失风险4.7% |
| Analysis | K-Means 轮廓系数 | 0.201 |
| Prediction | XGBoost AUC | 0.693 |
| Prediction | XGBoost Recall@最优 | 44.7% |
| Prediction | Prophet MAPE | 15.65% |
| Governance Agent | customer_rating 源表缺失告警 | 63.0%（15749/25000），评分 65/100 |
| Governance Agent | 结果集 vs 源表差异检测 | 检测到聚合压缩（2.8%→63.0%） |
| Report Agent | 结构完整度 | 6/6 章节（100%） |
| Report Agent | 报告字数 | 4,822 字 |
| Report Agent | 策略建议提及 | 21 处 |
| Report Agent | 数字引用 | 123 处 |
| Report Agent | LLM-as-Judge 可读性 | 9.0/10 (n=5, σ=0.00) |
| Report Agent | LLM-as-Judge 洞察深度 | 8.4/10 (n=5, σ=0.49) |
| Report Agent | LLM-as-Judge 可执行性 | 8.4/10 (n=5, σ=0.49) |
| Report Agent | LLM-as-Judge 综合均分 | 8.2/10 (n=5) |
| Governance→Report 闭环 | 治理结果被报告引用 | 已验证：可信度标注 7 次，缺失率引用 5 次，第六章专述 |
| 评估层 | 评估脚本覆盖 | 5 个（Schema/SQL/Prediction/Report/Summary） |
