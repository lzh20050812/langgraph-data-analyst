# 运行、故障与恢复手册

> 适用版本：阶段六，2026-09-19

## 状态与终态

正常路径为 `queued → running → completed|failed`；澄清路径为 `queued|running → waiting_clarification → queued`；取消路径为 `queued|waiting_clarification → cancelled` 或 `running → cancelling → cancelled`。`interrupted` 是兼容终态，表示嵌入式进程遗留任务或外部 Worker 租约重试耗尽。终态不可被迟到 Worker 改写。

所有通用状态写入均经过合法迁移校验。外部 Worker 领取任务时获得一次性 `execution_token`；事件、检查点、续租和终态提交都必须匹配当前令牌。租约交接会清空旧令牌并创建新令牌，因此即使旧 Worker 恢复，也不能写入结果。

## 重试与检查点

`POST /tasks/{id}/retry?mode=restart` 创建新任务，从头执行并保存 `parent_task_id`、`retry_mode=restart` 和重复风险列表。LLM 生成、只读 SQL 与下游分析都可能再次执行；配置了价格时也会再次计费。

`mode=checkpoint` 当前返回 409。检查点用于诊断和恢复依据，但 SQL、模型调用及部分分析节点还没有统一幂等键，直接从任意节点继续可能造成重复执行、重复计费或跳过前置校验。系统选择明确拒绝，而不是把“从头重跑”伪装为断点续跑。

## 故障分类与预算

- 网络、限流和超时：标记为可重试的提供方故障；单次模型请求由 SDK 的 `LLM_MAX_RETRIES` 控制。
- SQL、业务校验、预算和内部错误：任务级不自动重跑，进入 `failed`，由用户决定是否从头重试。
- 外部 Worker 中断：租约到期后在 `TASK_WORKER_MAX_ATTEMPTS` 内重新排队；超过预算进入 `interrupted`。
- 任务预算：限制 LLM 调用次数、单次上下文字符数、累计 Token、任务时长及估算费用。时长在节点和 LLM 调用边界协作检查，单次模型请求另受客户端超时约束；费用限制只有配置非零输入/输出单价时才有意义。

## 排障步骤

1. 查看 `/health/ready` 和 `/operations/summary`，确认数据库、模型配置、队列、Worker、Token 与失败分类。
2. 查看任务事件与最新检查点；事件不包含 prompt、query、认证信息，长字段会截断。
3. 外部模式检查 `/operations/workers`。租约未到期时不要人工复制任务；等待自动续租或交接。
4. 对 `failed`/`interrupted` 使用 `mode=restart`，并接受响应中的重复执行告警。不要手工把数据库状态改回 `queued`。
5. 若同一客户端请求可能重发，使用稳定的 `Idempotency-Key`；相同租户和请求体只会创建一个任务。

## 部署建议

单进程演示可用 `TASK_EXECUTION_MODE=embedded`。需要 API 与执行隔离时使用 Compose 的 `external` 模式和独立 Worker。任务 SQLite 文件必须位于可靠的单主机磁盘，不能由多主机同时共享。备份时同时复制 `tasks.db` 及其 WAL/SHM，或先停写后使用 SQLite backup API。

恢复演练至少覆盖：Worker 中断及租约交接、模型超时、重复提交、客户端断连、带 `Last-Event-ID` 的 SSE 重连。客户端断连只停止事件流，不取消任务；重连从最后序号继续读取。
