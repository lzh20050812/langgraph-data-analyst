# Runtime Architecture Upgrade

## Objective

This upgrade moves the project from a synchronous single-process demo toward a
recoverable analysis-task platform while preserving the original `/query`
contract and all evaluation entry points.

## Runtime flow

```text
Dashboard / API client
        |
        | POST /tasks
        v
FastAPI task API ---- SSE progress ----> Dashboard
        |
        v
Embedded executor or external leased worker
        |
        v
LangGraph workflow + node tracing
        |                 |
        |                 +--> durable node checkpoints
        +--------------------> durable events and final result
                                  |
                                  v
                         SQLite local task store
```

The local SQLite store uses WAL mode and one connection per operation. It is a
deliberate zero-service default for a thesis demonstration, not a claim that
SQLite is the final distributed queue. The storage boundary is isolated in
`api/task_store.py`; a multi-host deployment should first move authoritative
task metadata to PostgreSQL and preserve conditional state transitions and
execution-token fencing. A queue is introduced only after measurement proves
distribution latency is the bottleneck; Redis is not a second task truth source.

Tasks can be cancelled cooperatively. A cancellation request is observed at a
workflow node boundary, so the current database or model call finishes safely
and the most recent successful checkpoint remains inspectable. Terminal records
are cleaned by age and maximum-count policies.

## Identity and tenant isolation

When authentication is enabled, each credential resolves to a `Principal`
containing a stable subject and an `analyst` or `admin` role. New tasks persist
the subject as `owner_id`. Analysts can read and mutate only their own tasks,
events, checkpoints, reports, charts, summaries, and Prometheus metrics;
cross-tenant lookups return 404 to avoid disclosing object existence. Admins
receive the global operational view. Existing SQLite databases are migrated by
adding `owner_id='local'` without discarding task history.

`API_KEYS` remains compatible and maps every legacy key to a separate analyst
subject derived from a one-way key fingerprint. New deployments should use
`API_PRINCIPALS_JSON` to assign explicit identities and roles.

## External worker mode

`TASK_EXECUTION_MODE=external` separates HTTP request handling from model and
analytics execution. The API persists a queued task and returns immediately.
Workers claim the oldest task inside a SQLite `BEGIN IMMEDIATE` transaction, so
concurrent workers cannot claim the same row. A claim records `worker_id`, lease
expiry, and attempt number.

While processing, a daemon heartbeat renews the lease even during a long model
or database call. Another worker may requeue an expired lease; after
`TASK_WORKER_MAX_ATTEMPTS`, the task becomes interrupted rather than retrying
forever. Cancellation requests on abandoned leases resolve to cancelled. The
worker inventory is administrator-only, while aggregate Prometheus metrics can
remain tenant scoped.

## Retry-safe submission and durable audit

`POST /tasks` accepts an optional `Idempotency-Key`. The task store enforces a
partial unique index on `(owner_id, idempotency_key)`, so keys are isolated by
tenant and concurrent retries cannot enqueue duplicate analysis work. A SHA-256
fingerprint binds the key to the normalized query and intent. An exact retry
returns the original task; different content with the same key fails with HTTP
409. Neither the raw key nor the query is copied into the audit log.

Accepted, replayed, rejected, retried, and cancelled mutations are persisted in
`audit_events`. The administrator-only `/operations/audit` endpoint supports
sequence-based pagination and optional tenant filtering. Retention and
maximum-record limits keep this local audit store bounded.

## Shared tenant rate limits

Both `POST /query` and `POST /tasks` consume one allowance from the same
per-tenant analysis budget. Counters live in SQLite rather than process memory,
and each update is serialized with `BEGIN IMMEDIATE`; multiple API processes
therefore cannot independently oversubscribe the configured limit. Fixed
windows are removed after expiry, while rejections enter the durable audit
trail. Responses expose standard quota and reset headers, and administrators
can inspect active windows at `/operations/rate-limits`.

## Defense preflight

The administrator-only `/operations/preflight` endpoint runs a read-only
demonstration readiness audit. Critical checks cover populated allowlisted
tables, LLM configuration, SQLite integrity, an available embedded executor or
external Worker, and the dashboard asset. Advisory checks cover least-privilege
database grants, frozen evaluation artifacts, and the local Schema vector
store. Responses expose counts and status categories but never business rows,
credentials, grant strings, or raw provider errors.

## Contracts and traceability

- `TaskPlan`, `EvidenceBundle`, and `NodeTrace` are Pydantic contracts.
- Each workflow node emits start, completion, failure, and duration events.
- A bounded state snapshot is saved after every successful node. Large query
  and evidence row collections are capped in checkpoints.
- LLM events contain model name, duration, and token counts. Prompts, query
  rows, credentials, and raw provider errors are not included in those events.
- `LLM_FALLBACK_MODEL` optionally selects a second model after the configured
  client retry policy is exhausted.

## SQL defense in depth

Generated SQL is protected by four independent controls:

1. SQLGlot parses the statement with the MySQL dialect and permits one query
   expression only.
2. Write, DDL, export, command, transaction, and side-effect function nodes are
   rejected, and referenced tables must be in `SQL_ALLOWED_TABLES`.
3. The executed query receives a top-level `LIMIT` capped by
   `SQL_MAX_RESULT_ROWS`, and the driver uses bounded `fetchmany`.
4. MySQL executes under the dedicated `SELECT`-only account with a session
   execution timeout.

The legacy `sqlparse` and lexical checks remain as a second parser backstop and
for compatibility with the existing robustness evaluation.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `TASK_DB_PATH` | `./data/runtime/tasks.db` | Durable local task database |
| `TASK_EVENT_POLL_SECONDS` | `0.25` | SSE database polling interval |
| `TASK_STALE_SECONDS` | `300` | Heartbeat age before a task is interrupted |
| `MAX_QUEUED_TASKS` | `20` | Maximum waiting tasks before HTTP 429 |
| `API_RATE_LIMIT_REQUESTS` | `120` | Shared analysis requests per tenant/window |
| `API_RATE_LIMIT_WINDOW_SECONDS` | `60` | Shared fixed-window duration |
| `TASK_RETENTION_SECONDS` | `604800` | Terminal task retention period |
| `TASK_MAX_RECORDS` | `1000` | Maximum retained terminal tasks |
| `AUDIT_RETENTION_SECONDS` | `2592000` | Mutation audit retention (30 days) |
| `AUDIT_MAX_RECORDS` | `10000` | Maximum retained audit records |
| `TASK_EXECUTION_MODE` | `embedded` | In-API executor or external worker |
| `TASK_WORKER_POLL_SECONDS` | `0.5` | Empty-queue polling interval |
| `TASK_WORKER_LEASE_SECONDS` | `60` | Worker claim lease and heartbeat window |
| `TASK_WORKER_MAX_ATTEMPTS` | `2` | Crash-recovery execution budget |
| `SQL_MAX_RESULT_ROWS` | `1000` | Hard database result-row cap |
| `SQL_ALLOWED_TABLES` | project analytics tables | Query table allowlist |
| `LLM_FALLBACK_MODEL` | empty | Optional provider-compatible fallback model |
| `API_AUTH_ENABLED` | `false` | Require an API key on protected endpoints |
| `LOCAL_ACCESS_ENABLED` | `false` | Explicit trusted single-user authentication bypass |
| `API_KEYS` | empty | Comma-separated accepted API keys |
| `API_PRINCIPALS_JSON` | empty | Identity, role, and key records as JSON |

In Docker Compose, the API and worker mount the same `task_runtime_data` named
volume at `/app/data/runtime`. This keeps SQLite WAL on Docker's Linux
filesystem instead of a Windows bind mount. Pytest overrides `TASK_DB_PATH`
before application imports and uses a session-scoped temporary database, so a
local regression run cannot mutate or lock the live task store.

## Remaining production boundary

Before public deployment, add an external identity provider, TLS at the edge,
tenant-aware table/column policies, a shared transactional task store, and—if
measured demand requires it—a distributed queue. These controls are intentionally
separate from the local demonstration path so they cannot silently weaken or
complicate the reproducible thesis evaluation.
