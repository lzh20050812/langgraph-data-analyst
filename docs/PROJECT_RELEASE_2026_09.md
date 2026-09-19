# Project release 2026.09

该版本完成升级方案阶段一至阶段七的可执行工程工作：可靠性、多轮语义、双知识 RAG、SQL/证据校验、版本化评测、运行工程和产品展示。

## 版本标识

- 初始发布实现提交：`24386dd10664c53a853e4abee31fc2e38b4e2850`
- 本地标签：`v2026.09`
- 标签在最终验证后指向包含构建修复与验收记录的完整收尾提交；未执行远端推送或外网发布。
- 未跟踪目录 `research-proposal/` 不属于本版本，未纳入提交。

## 冻结入口

- 总体实现记录：`docs/UPGRADE_PROGRESS.md`
- 最终交付：`docs/FINAL_DELIVERY.md`
- 演示指南：`docs/DEFENSE_DEMO.md`
- 设计决策：`docs/KEY_DESIGN_DECISIONS.md`
- 评测协议：`docs/EVALUATION_PROTOCOL_V2.md`
- 基准边界：`docs/BENCHMARKS.md`

## 当前验收值

- Python 回归：217 passed（2026-09-19；另有 1 条第三方弃用警告）。
- 目录审计：179 条样本。
- 无模型确定性记录：31 条，全部通过。
- 前端：类型检查和 Vite 生产构建通过。
- Docker：当前代码镜像构建通过；App、Worker、MySQL 均健康，数据库初始化容器正常退出，`/health/live`、`/health/ready` 和前端入口返回 200。

数字以 `docs/UPGRADE_PROGRESS.md` 的最后一次实际执行记录为准；历史冻结论文结果不由本发布覆盖。
