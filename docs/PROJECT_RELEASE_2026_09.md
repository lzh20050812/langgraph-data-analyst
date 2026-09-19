# Project release 2026.09

该版本完成升级方案阶段一至阶段七的可执行工程工作：可靠性、多轮语义、双知识 RAG、SQL/证据校验、版本化评测、运行工程和产品展示。

## 版本标识

- 初始发布实现提交：`24386dd10664c53a853e4abee31fc2e38b4e2850`
- 初始发布标签：`v2026.09`（已发布；首次远端 CI 暴露第二 Schema 合成夹具未入库）
- 当前修订标签：`v2026.09.1`（补齐可复现 CI 所需的固定合成夹具）
- `main` 与发布标签已推送到 `origin`；保留 `v2026.09` 不改写，由 `v2026.09.1` 标记修复后的可复现版本。
- 未跟踪目录 `research-proposal/` 不属于本版本，未纳入提交。

## 冻结入口

- 总体实现记录：`docs/UPGRADE_PROGRESS.md`
- 最终交付：`docs/FINAL_DELIVERY.md`
- 演示指南：`docs/DEFENSE_DEMO.md`
- 设计决策：`docs/KEY_DESIGN_DECISIONS.md`
- 评测协议：`docs/EVALUATION_PROTOCOL_V2.md`
- 基准边界：`docs/BENCHMARKS.md`

## 当前验收值

- Python 回归：220 passed（2026-09-20；另有 1 条第三方弃用警告）。
- 目录审计：179 条样本。
- 无模型确定性记录：31 条，全部通过。
- 前端：类型检查和 Vite 生产构建通过。
- Docker：当前代码镜像构建通过；App、Worker、MySQL 均健康，数据库初始化容器正常退出，`/health/live`、`/health/ready` 和前端入口返回 200；完整发布验证并行执行期间 Worker 重启计数保持 0。
- Docker 一键验收：`scripts/verify_docker_release.ps1` 的完整构建模式与 `-SkipBuild` 快速模式均通过。
- 密钥文件边界：Compose 仅通过 `env_file` 注入进程环境，不再把宿主机 `.env` 文件挂载进容器；App、Worker、db-init 的 `/app/.env` 挂载数均为 0。
- 依赖与密钥审计：本地及容器 `pip check` 均无冲突；前端生产依赖经 npm 官方 registry 审计为 0 漏洞；已跟踪源码的高置信密钥模式扫描无命中。
- CI 可复现性：第二 Schema 的两份固定合成 CSV 已显式纳入版本控制；其他 `data/raw/*.csv` 继续保持忽略。

数字以 `docs/UPGRADE_PROGRESS.md` 的最后一次实际执行记录为准；历史冻结论文结果不由本发布覆盖。
