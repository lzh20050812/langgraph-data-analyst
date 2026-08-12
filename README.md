# AI Data Analyst —— 基于大语言模型的多智能体企业智能运营分析平台

> Production-ready · Reproducible · Evidence-grounded

基于 **LangGraph 的动态多智能体编排**，解决企业级 **Text2SQL 与运营分析的协同问题**。

## 当前优化主线：问题驱动的可追溯证据链

系统不再把每个问题都固定送入全部分析和预测模型。Planner 先生成结构化
`task_plan`，只调度问题真正需要的工具；SQL Agent 产生的查询、结果行和字段信息
进入共享 `EvidenceBundle`，随后由治理、分析、预测和报告节点持续补充。最终报告的
数字只能来自这条证据链，证据不足时必须明确说明，不能自行补造。

```text
用户问题 → task_plan → Schema/SQL → EvidenceBundle
                                 ├→ 数据治理
                                 ├→ 按需分析/预测
                                 └→ 证据约束报告
```

这一版还将显式 `intent` 接入路由，并把流失模型的阈值选择从测试集移到验证集。

SQL 链路增加了可审计的业务语义层：高频指标先绑定来源表、粒度、公式和结果列契约，
命中配方时不再让 LLM 猜测字段；无法合法连接时回退到可回答的单表/有效键查询。
未命中的开放问题仍进入 Schema RAG + LLM 自修正流程。MySQL 只允许单条只读
SELECT/CTE，并设置 60 秒执行上限。
所有准确率和模型指标均来自固定测试集与可复现实验脚本；指标只描述当前数据快照与评测边界，
不外推为开放域性能。

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
│   ├── task_planning.py      # 问题驱动的结构化任务计划
│   ├── evidence.py           # 跨节点共享 EvidenceBundle
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
- **为什么 SQL Agent 要加 SELECT 防御？** LLM 生成的 SQL 不能直接信任，这是生产数据系统的基本安全要求。系统使用 sqlparse + 正则双重校验，拒绝 DROP/DELETE/UPDATE 等危险操作。
- **为什么把 LLM 调用抽象成独立层？** 支持不同模型的可控基准对比，无需修改各 Agent 的业务逻辑。

## 验证状态

正式评测基于固定数据快照与测试集，保留逐样本结果、统计检验和复现信息。以下指标只描述当前评测边界。

| 评测项 | 结果 | 说明 |
|---|---:|---|
| 数据治理 | 缺失单元 1,176→0；重复问题 604→0 | 8,604 条输入记录，输出 8,000 个唯一客户 |
| Schema 检索 | BGE Field R@5 84.58%；字符 TF-IDF 90.00% | 保留负结果，不夸大向量检索收益 |
| Text2SQL | RAG 内容执行准确率 87.76%；完整 Schema 79.59% | 49 道可评分题；McNemar p=0.21875 |
| 业务语义层消融 | 开启 100%；关闭 46.67% | n=15；McNemar p=0.0078 |
| 多智能体综合任务 | TCS 100%；SQL 100%；报告质量 92.4/100 | 固定 50 题测试集 |
| RFM / K-Means | k=2；Silhouette 0.3105；ARI 0.997 | 默认 k=4 被实验否定 |
| 客户流失预测 | XGBoost AUC 0.6777；Logistic 0.6453 | 差值置信区间跨 0 |
| 营收预测 | Prophet MAPE 15.65%；末值基线 14.36% | Prophet 未击败最强简单基线 |
| 历史分析记忆 | R@1 80%；R@3 90%；MRR@3 0.85 | 10 份记忆、10 条同义改写 |
| 鲁棒性 | OOD 6/6 fail closed；注入 4/4 隔离 | 核心表行数前后一致 |
| API 并发 | 1/2/4 并发成功率均 100% | 4 worker 吞吐 1.392 req/s |
| Docker | MySQL/App 双容器，`/health = ok` | 端到端部署已验证 |

回归测试基线：**32 passed**。

## 技术栈

Python · LangGraph · LangChain · MySQL · FastAPI · Pandas · ChromaDB · XGBoost · Prophet · ECharts · Docker · BAAI/bge-small-zh · DeepSeek-V3
