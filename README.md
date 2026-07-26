# AI Data Analyst —— 基于大语言模型的多智能体企业智能运营分析平台

> 本科毕业设计项目 · 简历/GitHub 展示项目

基于 **LangGraph 的动态多智能体编排**，解决企业级 **Text2SQL 与运营分析的协同问题**。

## 架构总览（七层）

```
数据源层 → ETL工程层 → 数据存储层 → Planner Agent → 多智能体协同层
  → 评估层 → 应用层 → 服务化与部署层
```

| 层级 | 组件 | 技术栈 |
|------|------|--------|
| 数据源层 | Kaggle CSVs → 模拟异构脏数据 | Pandas |
| ETL工程层 | 字段映射、清洗、去重、质量报告 | Python 脚本（非Agent） |
| 数据存储层 | MySQL + ChromaDB + Schema Metadata | SQLAlchemy, ChromaDB, BGE |
| Planner Agent | LangGraph 状态图入口 | LangGraph |
| 多智能体层 | Schema / SQL / 数据治理 / Analysis / Prediction / Report | 6 Agents |
| 评估层 | Schema检索准确率、SQL成功率、模型指标 | 独立评估脚本 |
| 应用层 | FastAPI + ECharts 看板 | FastAPI, ECharts |

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，填入你的 LLM API Key：

```env
LLM_API_KEY=your_deepseek_api_key_here
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 3. 启动 MySQL（Docker）

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 4. 一键初始化

```bash
# 生成脏数据 + ETL + 导入 MySQL + 创建 ChromaDB 向量库
python -m storage.init_storage
```

### 5. 运行评估

```bash
python -m evaluation.eval_schema       # Schema Agent 检索准确率
python -m evaluation.eval_sql          # SQL Agent 生成成功率
python -m evaluation.eval_prediction   # 模型评估（XGBoost + Prophet）
python -m evaluation.eval_report       # Report Agent 可读性评估
python -m evaluation.eval_summary      # 总评估汇总
```

### 6. 启动应用

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# 打开 http://localhost:8000 查看可视化看板
```

### 7. Docker 一键部署

```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

## 目录结构

```
├── data/
│   ├── raw/                  # 原始数据 + 脏数据 + injection_report.json
│   └── processed/            # 清洗后数据 + 评估报告
├── scripts/
│   └── generate_dirty_sources.py  # 脏数据生成（可重复，seed=42）
├── etl/                      # ETL工程层（纯Python，非Agent）
│   ├── field_mapper.py       # 字段名映射
│   ├── cleaner.py            # 去重+缺失值+格式统一
│   ├── loader.py             # MySQL写入
│   ├── quality_report.py     # 治理前后质量对比
│   └── run_etl.py            # ETL端到端验证
├── storage/                  # 数据存储层
│   ├── mysql/                # MySQL（schema.sql, client）
│   ├── chromadb/             # ChromaDB（schema_metadata, embedder）
│   └── init_storage.py       # 一键初始化
├── agents/                   # 多智能体协同层
│   ├── state.py              # AgentState 定义
│   ├── llm.py                # LLM调用抽象层
│   ├── planner.py            # LangGraph 状态图
│   ├── schema_agent.py       # Schema Agent
│   ├── sql_agent.py          # SQL Agent（含自修正+SELECT防御）
│   ├── governance_agent.py   # 数据治理Agent（Phase 3）
│   ├── analysis_agent.py     # Analysis Agent（Phase 2）
│   ├── prediction_agent.py   # Prediction Agent（Phase 2）
│   ├── report_agent.py       # Report Agent（Phase 3）
│   └── chart_renderer.py     # 图表渲染
├── evaluation/               # 评估层
│   ├── eval_schema.py        # Schema检索准确率
│   ├── eval_sql.py           # SQL成功率+重试曲线
│   ├── eval_prediction.py    # 模型评估（Phase 2）
│   └── eval_report.py        # 报告可读性（Phase 3）
├── api/                      # FastAPI应用层（Phase 3）
├── frontend/                 # ECharts可视化（Phase 3）
├── docker/                   # Docker部署
├── config/                   # 全局配置
│   ├── settings.py           # 配置入口
│   └── prompts/              # LLM Prompt模板
├── notebooks/                # Jupyter演示（Phase 2+）
├── requirements.txt
└── README.md
```

## 六个 Agent 说明

| Agent | 职责 | 实现深度 |
|-------|------|----------|
| **Schema Agent** | ChromaDB语义检索 + LLM精排，业务术语→表字段映射 | 重点实现 |
| **SQL Agent** | NL→SQL→执行→自修正闭环 + SELECT安全校验 | 重点实现 |
| **数据治理Agent** | SQL查询结果运行时质量评分（如"customer_rating缺失63%，可信度低"） | 简化实现 |
| **Analysis Agent** | RFM客户价值分析、K-Means聚类、运营指标计算 | 重点实现 |
| **Prediction Agent** | XGBoost流失预测 + Prophet销售预测 | 重点实现 |
| **Report Agent** | LLM生成经营洞察 + 策略建议 + 结构化报告 | 重点实现 |

## 设计决策

- **为什么 ETL 只清洗 customers 表？** orders 表的 `customer_rating` 天然缺失约63%，保留 NULL 供数据治理Agent 在运行时评分，展示真实数据质量场景。
- **为什么不用模糊匹配去重？** App端"近似重复"本质上同一 customerId 出现两次，直接按ID保留首条；Web端完全重复直接 drop_duplicates()。避免过度设计。
- **为什么 SQL Agent 要加 SELECT 防御？** LLM生成的SQL不能直接信任，这是面试/答辩必问的安全问题。使用 sqlparse + 正则双重校验，拒绝 DROP/DELETE/UPDATE 等危险操作。
- **为什么把 LLM 调用抽象成独立层？** 方便后续做"不同 LLM 下 SQL 生成准确率对比"加分实验，不用修改各 Agent 代码。

## 完成状态

### Phase 1：核心链路（数据 → ETL → 存储 → Text2SQL）

- [x] 脏数据生成（App/Web两个异构源 + injection_report.json，seed=42可复现）
- [x] ETL 完整流程（字段映射 → 清洗 → 去重 → 质量对比，缺失值/重复行 100% 修复）
- [x] MySQL Schema DDL（4张表 + 日/月粒度聚合视图）
- [x] ChromaDB Schema 向量库（54条字段元数据，BAAI/bge-small-zh）
- [x] Planner Agent（LangGraph 状态图，条件路由）
- [x] Schema Agent（ChromaDB 粗排 + LLM 精排，Top-5 命中率 **95%**）
- [x] SQL Agent（生成→安全校验→执行→自修正闭环，可解题首次成功率 **95.8%**）

### Phase 2：核心分析能力（Analysis + Prediction）

- [x] Analysis Agent（RFM 客户分段 + K-Means 聚类 + 运营指标）
- [x] Prediction Agent（XGBoost 流失预测 AUC **0.693** + Prophet 销售预测 MAPE **15.65%**）
- [x] Chart Renderer（分析/预测结果 → ECharts 配置 JSON）

### Phase 3：收尾与工程化（Report + Governance + API + Docker）

- [x] Governance Agent（运行时数据质量评分，含源表追溯层，customer_rating 63%缺失 → 65/100）
- [x] Report Agent（LLM 经营洞察报告，LLM-as-Judge 综合 **8.2/10**（n=5），可读性 **9.0/10**）
- [x] FastAPI 应用层（`POST /query` + `GET /charts` + ECharts 可视化看板）
- [x] Docker 部署（MySQL + FastAPI 双容器，ChromaDB 嵌入+Volume 持久化）
- [x] 评估层汇总（5 个评估脚本覆盖全模块）

## 技术栈

Python · LangGraph · LangChain · MySQL · FastAPI · Pandas · ChromaDB · XGBoost · Prophet · ECharts · Docker · BAAI/bge-small-zh · DeepSeek-V3
