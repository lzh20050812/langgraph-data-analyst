# AI Data Analyst —— 基于大语言模型的多智能体企业智能运营分析平台

本文档是项目开发的总纲，请在整个开发过程中以此为准。如果后续实现和本文档冲突，先停下来跟我确认，不要自行改变架构定位。

## 一、项目定位

这是一个本科毕业设计项目，同时要作为简历项目和 GitHub 展示项目使用。核心叙事是：**基于 LangGraph 的动态多智能体编排，解决企业级 Text2SQL 与运营分析的协同问题**，而不是"调用几次 LLM API 拼起来的分析脚本"。

请在写代码时始终记住三个约束：
1. **范围要可控**——这是一个人在有限时间内完成的项目，不是企业级产品，遇到"要不要做得更复杂"的选择时，优先选择能跑通、可验证的版本。
2. **Agent 和工程脚本要分清楚**——凡是确定性规则处理（清洗、去重、格式转换）一律写成普通 Python 函数/脚本，不要包装成"Agent"；只有真正需要 LLM 做语义理解、规划、生成的环节才是 Agent。
3. **每个模块都要能单独验证效果**——不要等所有模块都写完才想起来评估，每完成一个 Agent 就顺手写好对应的评估脚本或测试用例。

## 二、技术栈

Python、LangGraph、LangChain、MySQL、FastAPI、Pandas、ChromaDB、XGBoost、Prophet、ECharts、Docker

## 三、系统架构（七层，自底向上）

```
数据源层
  → ETL工程层（批处理脚本，非Agent）
  → 数据存储层（MySQL / ChromaDB / Schema Metadata）
  → Planner Agent（LangGraph 状态图入口）
  → 多智能体协同层（Schema / SQL / 数据治理 / Analysis / Prediction / Report 六个 Agent，共享 AgentState）
  → 评估层（模型评估 / 产出质量评估 / 系统评估）
  → 应用层（FastAPI 提供接口，前端用 ECharts 做可视化看板 + 自然语言问答 + 预测展示）
  → 服务化与部署层（FastAPI + Docker）
```

### 3.1 数据源层

原始数据来自 Kaggle，位于 `data/raw/`，一共 4 个干净的 CSV，**注意：原始数据本身是干净的，不含缺失值和重复记录**，多源异构和数据质量问题需要你自己在 customers.csv 基础上人为构造出来：

- `customers.csv`（8000行20列）：customer_id, country, age, gender, membership_tier, registration_date, total_orders, total_spend_usd, avg_order_value_usd, days_since_last_purchase, preferred_category, preferred_device, preferred_payment_method, acquisition_channel, reviews_given, avg_review_score, returns_made, wishlist_items, newsletter_subscribed, churned
- `orders.csv`（25000行28列）：核心交易明细表，**customer_rating 字段天然有约63%缺失**，这是数据集自带的真实缺失，不需要额外注入，直接用来验证数据治理Agent对"自然缺失"的识别效果即可
- `monthly_revenue.csv`（75行，按月聚合）：year, month, quarter, orders, revenue_usd, avg_order_value, avg_discount_pct, return_rate, unique_customers, new_customers，用于 Prophet 销售预测
- `product_summary.csv`（140行，按商品聚合）：category, product_name, total_orders, total_revenue_usd, avg_price, avg_rating, return_rate, avg_discount_pct, avg_delivery_days

**请写一个独立的脏数据生成脚本 `scripts/generate_dirty_sources.py`，把干净的 customers.csv 拆分并污染成两个模拟的异构来源，规则如下（要求可重复运行、带随机种子，方便论文里写清楚注入比例）：**

1. **拆分成两个来源**：按 customer_id 哈希或随机抽样，切出 `customers_app_raw.csv`（约60%客户）和 `customers_web_raw.csv`（约40%客户），两者故意保留 5% 左右的 customer_id 交集，模拟同一用户在两端都注册过。

2. **customers_app_raw.csv**（模拟App端导出，字段名走驼峰/缩写风格）：
   - 列重命名：`customer_id→customerId`，`registration_date→regDate`，`preferred_device→prefDevice`，`preferred_payment_method→payMethod`，`acquisition_channel→channel`，其余列名不变
   - `gender` 字段编码改成 `M`/`F`（而不是原来的 Male/Female）
   - `regDate` 日期格式改成 `MM/DD/YYYY`
   - 随机对 `age`、`avg_review_score` 两列注入约 6% 缺失值（置空）
   - 随机复制约 2% 的行作为重复记录（部分字段有细微差异，比如 total_spend_usd 相差几分钱，模拟重复上报）

3. **customers_web_raw.csv**（模拟Web端导出，字段名走snake_case但和原始命名不完全一致）：
   - 列重命名：`customer_id→cust_id`，`registration_date→signup_date`，`preferred_category→fav_category`，`newsletter_subscribed→subscribed_flag`，其余列名不变
   - `gender` 字段保留 Male/Female，但大小写不统一（部分行是 `male`/`FEMALE`）
   - `signup_date` 日期格式改成 `YYYY年MM月DD日` 或 ISO 格式混用（模拟两次系统迁移遗留的格式不一致）
   - 随机对 `country`、`preferred_device`（此处列名是 fav_device）两列注入约 8% 缺失值
   - 随机制造约 3% 的重复记录（完全重复的行，模拟导出脚本重跑导致的重复）

4. 生成脚本要同时输出一份 `data/raw/injection_report.json`，记录每个来源、每一列的缺失值注入数量和比例、重复行数量，这份报告后面直接可以用在论文"数据质量问题描述"和数据治理Agent的效果验证对比里（治理前 vs 治理后）。

ETL 工程层要能把这两个异构来源重新映射回统一 schema（对应 3.2 节的"字段映射清洗"），最终写回 MySQL 时是一张干净统一的 customers 表。

### 3.2 ETL工程层（非Agent，普通脚本）
- 字段映射清洗：把 `customers_app_raw.csv` 和 `customers_web_raw.csv` 的字段名（如 `customerId`/`cust_id`）、日期格式、性别编码统一映射回标准 schema（对齐原始 customers.csv 的列名规范）
- 去重与缺失处理：处理 3.1 节中注入的重复记录和缺失值，处理逻辑和结果要能和 `injection_report.json` 对比，量化"治理前后"的数据质量提升
- 写入 MySQL：清洗后的 customers 表 + 原样导入的 orders、monthly_revenue、product_summary

### 3.3 数据存储层
- MySQL：customers（清洗后统一表）、orders、monthly_revenue、product_summary 四张业务表
- 建议在 orders 表所在库里加一张视图或统计表用于日粒度/月粒度聚合，供 Prediction Agent 直接查询，避免每次都全表扫描 25000 行订单明细
- ChromaDB：存储表结构、字段语义的 embedding，供 Schema Agent 检索
- Schema Metadata：结构化的表/字段元信息（人类可读的业务术语映射表）

### 3.4 Planner Agent
基于 LangGraph 状态图实现，负责：
- 解析用户自然语言需求，判断需要哪些下游 Agent
- 维护并向下游 Agent 传递共享状态 `AgentState`（不是知识库，是一次任务执行过程中的运行时状态，字段包括 user_query / selected_tables / sql / query_result / analysis_result / prediction_result / charts / report）
- 支持条件路由（不是固定顺序执行，某些任务可能不需要 Prediction Agent）

### 3.5 多智能体协同层（六个 Agent）

| Agent | 职责 | 实现深度 |
|---|---|---|
| Schema Agent | 基于 ChromaDB 做语义检索，把用户问题里的业务术语（如"GMV下降"）映射到具体的表和字段 | 重点实现，需要有检索准确率的评估 |
| SQL Agent | 自然语言生成 SQL → 执行 → 若执行失败，把报错信息喂回 LLM 重新生成（自修正闭环） | 重点实现，需要有 SQL 执行成功率的评估 |
| 数据治理Agent | 对某次查询返回的数据做运行时质量评分和异常标记（不是批处理清洗，是轻量级的"这批数据靠谱吗"判断） | 简化实现，能跑出一个质量分数即可，不用做复杂的异常检测算法 |
| Analysis Agent | RFM 客户价值分析、K-Means 客户聚类、GMV/客单价/复购率等运营指标计算 | 重点实现，需要有实际数据集上的结果和指标 |
| Prediction Agent | XGBoost 客户流失预测、Prophet 销售趋势预测 | 重点实现，需要有 AUC/F1（流失预测）和 RMSE/MAPE（销售预测）等指标 |
| Report Agent | 综合前面所有 Agent 的输出，用 LLM 生成经营洞察 + 策略建议 + 结构化报告（不只是罗列数字，要给出"为什么"和"怎么办"） | 重点实现，需要有人工评分/可读性的评估方式 |

可视化（ECharts 图表）不做成独立 Agent，作为 Analysis / Prediction / Report Agent 输出结果之后的一个渲染步骤即可。

### 3.6 评估层（不要省略，这是毕设答辩必问的部分）
- 模型评估：AUC/F1（流失预测）、RMSE/MAPE（销售预测）、聚类内部评估指标
- 产出质量评估：SQL 生成准确率、报告可读性人工评分
- 系统评估：接口响应时间、任务完成率

### 3.7 应用层与部署
- FastAPI 暴露接口（对话式查询接口 + 图表数据接口）
- 前端展示：可视化看板、智能运营问答、预测结果展示（前端技术不限，能跑起来即可，不是本项目重点）
- Docker 完成容器化，保证环境一致性

## 四、开发优先级建议（分阶段，避免范围失控）

**第一阶段（打通主链路，最优先）**
数据源模拟 → ETL 脚本 → MySQL 建表入库 → Planner + Schema Agent + SQL Agent 打通"自然语言问数据库"这条链路，先不接 Analysis/Prediction。

**第二阶段（核心分析能力）**
Analysis Agent（RFM + K-Means）、Prediction Agent（XGBoost 流失预测、Prophet 销售预测），配套写评估脚本，产出真实的指标数字。

**第三阶段（收尾与工程化）**
Report Agent、数据治理Agent（简化版即可）、评估层汇总、FastAPI 封装、Docker 部署、前端可视化。

**明确不做/降低优先级的事**：不做多租户、不做用户权限系统、不做复杂的 Agent 间投票/仲裁机制、不追求 SQL Agent 支持所有 SQL 语法边界情况。

## 五、交付物要求

- 清晰的目录结构（建议按上面七层拆分模块目录，比如 `etl/`、`agents/`、`storage/`、`evaluation/`、`api/`）
- 每个 Agent 独立可测试，有对应的评估脚本或 notebook
- README 要能直接对应本文档的架构图，方便简历/面试讲解
- 关键设计决策（比如为什么数据治理不算 ETL 层的一部分）在代码注释或文档里体现，方便写论文时直接引用

## 六、请先做的事

在开始写代码前，先输出一份目录结构规划和第一阶段的具体任务拆解，跟我确认后再开始实现。
