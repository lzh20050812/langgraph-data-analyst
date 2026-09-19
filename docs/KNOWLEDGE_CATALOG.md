# 知识目录与更新流程

## 边界

阶段三把知识分成两类，运行时分别检索、分别展示来源：

- 结构知识：数据源、表、字段、类型、表关系和基数，定义在 `storage/chromadb/schema_metadata.py`。
- 业务知识：指标名称、别名、公式、必要字段、支持维度和限制，定义在 `agents/analysis_request.py`，由 `storage/chromadb/business_metadata.py` 转成可检索条目。

样例 SQL 不属于权威知识。即使命中确定性配方，仍需经过当前允许表、只读 SQL、字段归属、请求范围和结果形状校验。

## 版本与来源

每条结构知识包含 `data_source`、`catalog_version`、`source_id`、`access_scope` 和 `owner_id`；表关系另有独立版本。业务指标包含指标版本和目录版本。`schema_catalog_fingerprint()` 对字段及关系生成内容指纹，Chroma collection 同时保存该指纹。

Schema Agent 和独立评测入口会比较当前目录指纹与索引元数据。集合不存在或指纹不一致时才重建，因此字段新增、删除、来源或权限变更不会继续使用旧索引。

可通过受认证接口检查当前目录：

```text
GET /knowledge/catalog?data_source=ai_analytics
GET /knowledge/catalog?data_source=support_ops
GET /metrics/catalog
```

接口只返回当前主体可见的条目。

## 权限规则

`access_scope=public` 对已认证主体可见；非公开条目必须满足 `owner_id == principal.subject`。词法检索在评分前过滤，Chroma 查询使用 `data_source` 与 public/owner 条件在集合查询层过滤，LLM 精排结果还会再次对照可见字段白名单；模型不能注入不存在或无权访问的字段。

## 关联路径

检索结果同时包含已知关联两端的表时，`complete_relationship_paths()` 自动补全两侧关联键，并标记为 `required_join`。当前受控关系：

- `orders.customer_id = customers.customer_id`（many-to-one）
- `support_tickets.agent_id = support_agents.agent_id`（many-to-one）

没有登记的关系不会由检索层猜测。

## 第二业务 Schema

`support_ops` 与电商数据共用 MySQL 数据库，但使用不同的表和业务语义：

- `support_agents`：客服团队与负责区域。
- `support_tickets`：工单渠道、优先级、状态、解决耗时与满意度。

固定小数据位于 `data/raw/support_agents.csv` 和 `data/raw/support_tickets.csv`。`python -m storage.init_storage --skip-etl` 会装载两表；默认 SQL 白名单已包含它们。该接入证明目录、检索、关联补全和 SQL 执行边界可按 `data_source` 迁移，但不代表任意外部 Schema 已实现零代码接入。

## 更新步骤

1. 在结构目录中新增或修改字段、表和显式关系，填写数据源、来源、版本及权限。
2. 在指标目录中更新受控口径和限制；不能把示例问题直接保存为业务定义。
3. 添加陌生表达、同名字段、关联路径和权限负例。
4. 运行 `python -m evaluation.experiments.hybrid_schema_evaluation`。参数只由验证集选择，测试集只用于最终报告。
5. 运行完整 pytest；若接入新表，再对目标数据库执行只读关联查询。

## 当前限制

- 权限粒度目前是 public 或精确 owner，尚无组织/部门层级 ACL。
- 第二 Schema 使用同一 MySQL 实例，尚未验证跨数据库方言或跨库连接。
- 独立检索集仍以电商 Schema 为主；客服 Schema 目前用确定性迁移回归覆盖，后续应扩充独立陌生表达测试集。
- 2026-09-18 重跑结果中 Hybrid Field Recall@5 为 86.31%，词法为 85.71%，差值 0.60 个百分点且 Bootstrap 95% CI 跨 0，不能宣称显著提升。
