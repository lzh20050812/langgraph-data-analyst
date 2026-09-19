"""
FastAPI 应用层 —— 对外暴露对话式查询接口 + 图表数据接口。

接口：
- POST /query    对话式自然语言查询（支持 sql_query / analysis / prediction / mixed 全链路）
- GET  /charts/{request_id} 获取指定查询生成的图表数据列表
- GET  /report/{request_id} 获取指定查询生成的报告
- GET  /health   健康检查
- GET  /         静态前端看板 (frontend/index.html)

启动方式：
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    或: python -m api.main
"""

import asyncio
from hashlib import sha256
import json
import logging
import sys
from pathlib import Path
from time import perf_counter, time
from typing import Literal, Optional
from uuid import uuid4

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, field_validator

from api.result_store import result_store
from api.preflight import PreflightReport, run_preflight
from api.portal_store import portal_store
from api.data_catalog import catalog as load_catalog, preview as load_preview, table_schema as load_table_schema
from api.demo_scenarios import build_demo_result, list_demo_scenarios
from api.query_limiter import query_limiter
from api.task_runtime import (
    SubmissionReceipt,
    TaskQueueFull,
    UnsafeCheckpointResume,
    task_runtime,
    task_store,
)
from api.task_store import IdempotencyConflict, TERMINAL_STATUSES
from config.settings import get_settings
from api.auth import LOCAL_PRINCIPAL, Principal, authenticate_api_key, configured_credentials, requires_auth
from agents.run_context import run_hooks
from agents.analysis_request import (
    CONFIRMED_REQUEST_PREFIX,
    build_analysis_request,
    inherit_analysis_request,
    metric_catalog_payload,
)
from storage.chromadb.business_metadata import business_knowledge_entries
from storage.chromadb.schema_metadata import (
    SCHEMA_CATALOG_VERSION,
    SCHEMA_RELATIONSHIPS,
    get_schema_fields,
    is_accessible,
    schema_catalog_fingerprint,
)

app = FastAPI(
    title="AI Data Analyst — 智能运营分析平台",
    description="基于 LangGraph 的多智能体企业运营分析 API",
    version="1.0.0",
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
if (FRONTEND_DIST_DIR / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST_DIR / "assets"),
        name="frontend-assets",
    )


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})[
        "ApiKeyAuth"
    ] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    for path, operations in schema.get("paths", {}).items():
        if requires_auth(path):
            for operation in operations.values():
                if isinstance(operation, dict):
                    operation["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

# Allow the bundled dashboard to call the local API even when index.html is
# opened directly from disk (browsers send the special ``null`` origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Idempotency-Key"],
    allow_credentials=True,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_stdout_handler)
logger.propagate = False


@app.middleware("http")
async def optional_api_key_auth(request: Request, call_next):
    """Resolve API-key or browser-session authentication into one Principal."""
    try:
        settings = get_settings()
        api_key = request.headers.get("X-API-Key")
        if api_key:
            principal = authenticate_api_key(api_key)
        elif getattr(settings, "WEB_LOGIN_ENABLED", False):
            principal = portal_store.principal_for_session(
                request.cookies.get("analytics_session")
            )
        elif getattr(settings, "LOCAL_ACCESS_ENABLED", True):
            # Tests and legacy local runs explicitly disable both login
            # mechanisms. Shared/browser deployments must opt in separately.
            principal = LOCAL_PRINCIPAL
        else:
            principal = None
    except (ValueError, json.JSONDecodeError):
        logger.error("API authentication configuration is invalid")
        return JSONResponse(
            {"detail": "API 认证配置无效"},
            status_code=503,
        )
    if (
        request.method != "OPTIONS"
        and requires_auth(request.url.path)
        and principal is None
    ):
        return JSONResponse(
            {"detail": "未登录或访问凭证无效"},
            status_code=401,
            headers={
                "WWW-Authenticate": (
                    "Session" if getattr(settings, "WEB_LOGIN_ENABLED", False)
                    else "ApiKey"
                )
            },
        )
    request.state.principal = principal or LOCAL_PRINCIPAL
    return await call_next(request)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    """Attach one server-generated correlation ID and duration to every request."""
    request_id = uuid4().hex
    request.state.request_id = request_id
    started = perf_counter()
    response = await call_next(request)
    duration_ms = round((perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    principal = getattr(request.state, "principal", None)
    log_record = {
        "event": "http_request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
    }
    if principal is not None:
        log_record["principal"] = principal.subject
        log_record["role"] = principal.role
    logger.info(json.dumps(log_record, ensure_ascii=False))
    return response

# ---- 请求模型 ----

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    intent: Optional[Literal["sql_query", "analysis", "prediction", "mixed"]] = None

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

class QueryResponse(BaseModel):
    request_id: str
    success: bool
    intent: str
    sql: Optional[str] = None
    query_result: Optional[list] = None
    query_result_total_rows: Optional[int] = None
    query_result_truncated: bool = False
    governance_result: Optional[dict] = None
    analysis_result: Optional[dict] = None
    prediction_result: Optional[dict] = None
    evidence: Optional[dict] = None
    report: Optional[str] = None
    charts: Optional[list] = None
    error: Optional[str] = None
    messages: Optional[list] = None
    execution_trace: Optional[list] = None


class TaskSubmission(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    intent: Optional[Literal["sql_query", "analysis", "prediction", "mixed"]] = None

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)
    display_name: str = Field(min_length=1, max_length=80)
    role: Literal["analyst", "admin"] = "analyst"


class UserUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    role: Literal["analyst", "admin"] | None = None
    enabled: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=256)


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=100)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value):
        return value.strip() if isinstance(value, str) else value


class ConversationMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1000)
    intent: Optional[Literal["sql_query", "analysis", "prediction", "mixed"]] = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value):
        return value.strip() if isinstance(value, str) else value


def _request_principal(request: Request) -> Principal:
    return getattr(request.state, "principal", LOCAL_PRINCIPAL)


def _task_for_principal(task_id: str, request: Request) -> dict:
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="分析任务不存在")
    principal = _request_principal(request)
    if not principal.is_admin and task.get("owner_id") != principal.subject:
        # Do not disclose whether another tenant's task exists.
        raise HTTPException(status_code=404, detail="分析任务不存在")
    return task


def _require_admin(request: Request) -> Principal:
    principal = _request_principal(request)
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return principal


def _record_audit(
    request: Request,
    *,
    action: str,
    resource_type: str,
    resource_id: str | None,
    outcome: str,
    metadata: dict | None = None,
) -> None:
    """Persist a safe mutation record without making audit failure user-facing."""
    principal = _request_principal(request)
    try:
        task_store.append_audit(
            owner_id=principal.subject,
            role=principal.role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=getattr(request.state, "request_id", None),
            outcome=outcome,
            metadata=metadata,
        )
        settings = get_settings()
        task_store.prune_audit(
            settings.AUDIT_RETENTION_SECONDS, settings.AUDIT_MAX_RECORDS
        )
    except Exception:
        logger.exception("Failed to persist audit event")


def _enforce_analysis_rate_limit(request: Request, response: Response) -> None:
    """Apply a SQLite-backed per-tenant budget shared by all API processes."""
    settings = get_settings()
    limit = settings.API_RATE_LIMIT_REQUESTS
    if limit == 0:
        return
    window_seconds = settings.API_RATE_LIMIT_WINDOW_SECONDS
    if limit < 0 or window_seconds < 1:
        raise HTTPException(status_code=503, detail="API 限流配置无效")
    principal = _request_principal(request)
    decision = task_store.consume_rate_limit(
        owner_id=principal.subject,
        route_scope="analysis",
        limit=limit,
        window_seconds=window_seconds,
    )
    headers = {
        "X-RateLimit-Limit": str(decision["limit"]),
        "X-RateLimit-Remaining": str(decision["remaining"]),
        "X-RateLimit-Reset": str(decision["reset_at"]),
    }
    for name, value in headers.items():
        response.headers[name] = value
    if not decision["allowed"]:
        _record_audit(
            request,
            action="api.rate_limit",
            resource_type="route",
            resource_id=request.url.path,
            outcome="rate_limited",
            metadata={
                "route_scope": "analysis",
                "limit": limit,
                "window_seconds": window_seconds,
            },
        )
        raise HTTPException(
            status_code=429,
            detail="请求频率超过租户配额，请稍后重试",
            headers={
                **headers,
                "Retry-After": str(decision["retry_after"]),
            },
        )


# ---- 接口 ----

@app.get("/health/live")
def liveness_check():
    """进程存活检查，不访问任何外部依赖。"""
    return {"status": "ok"}


def _readiness_payload() -> tuple[dict, int]:
    """检查数据库连通性和 LLM 配置完整性。"""
    health = {
        "status": "ok",
        "mysql": False,
        "llm": False,
        "checks": {
            "mysql": "unavailable",
            "llm": "not_configured",
        },
    }

    # MySQL
    try:
        from storage.mysql.client import check_connection
        health["mysql"] = check_connection()
        if health["mysql"]:
            health["checks"]["mysql"] = "connected"
    except Exception:
        pass

    # 就绪检查只验证配置，不向按量计费的外部 LLM 发请求。
    try:
        from config.settings import get_settings
        settings = get_settings()
        health["llm"] = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)
        if health["llm"]:
            health["checks"]["llm"] = "configured"
    except Exception:
        pass

    auth_valid = True
    try:
        settings = get_settings()
        auth_enabled = getattr(settings, "API_AUTH_ENABLED", False)
        auth_valid = not auth_enabled or bool(configured_credentials(settings))
        if auth_enabled:
            health["auth"] = auth_valid
            health["checks"]["auth"] = (
                "configured" if auth_valid else "not_configured"
            )
    except Exception:
        auth_valid = False

    worker_valid = True
    try:
        if getattr(settings, "TASK_EXECUTION_MODE", "embedded") == "external":
            active_workers = task_store.list_workers(
                active_within_seconds=max(5, settings.TASK_WORKER_LEASE_SECONDS)
            )
            worker_valid = bool(active_workers)
            health["worker"] = worker_valid
            health["checks"]["worker"] = (
                "connected" if worker_valid else "unavailable"
            )
    except Exception:
        worker_valid = False

    rate_limit_valid = True
    try:
        rate_limit_valid = (
            getattr(settings, "API_RATE_LIMIT_REQUESTS", 120) >= 0
            and getattr(settings, "API_RATE_LIMIT_WINDOW_SECONDS", 60) >= 1
        )
        if not rate_limit_valid:
            health["rate_limit"] = False
            health["checks"]["rate_limit"] = "invalid_configuration"
    except Exception:
        rate_limit_valid = False

    if (
        not health["mysql"]
        or not health["llm"]
        or not auth_valid
        or not worker_valid
        or not rate_limit_valid
    ):
        health["status"] = "degraded"

    return health, 200 if health["status"] == "ok" else 503


@app.get("/health/ready")
def readiness_check():
    """依赖就绪检查；未就绪时返回 HTTP 503。"""
    payload, status_code = _readiness_payload()
    return JSONResponse(payload, status_code=status_code)


@app.get("/health")
def health_check():
    """向后兼容的就绪检查别名。"""
    return readiness_check()


@app.get("/auth/config")
def authentication_config():
    settings = get_settings()
    return {
        "web_login_enabled": getattr(settings, "WEB_LOGIN_ENABLED", False),
        "api_key_enabled": settings.API_AUTH_ENABLED,
    }


@app.post("/auth/login")
def login(req: LoginRequest, request: Request):
    settings = get_settings()
    if not getattr(settings, "WEB_LOGIN_ENABLED", False):
        raise HTTPException(status_code=404, detail="浏览器登录未启用")
    login_identity = sha256(req.username.strip().lower().encode("utf-8")).hexdigest()[:16]
    decision = task_store.consume_rate_limit(
        owner_id=f"login-{login_identity}",
        route_scope="authentication",
        limit=5,
        window_seconds=300,
    )
    if not decision["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="登录尝试过多，请稍后重试",
            headers={"Retry-After": str(decision["retry_after"])},
        )
    user = portal_store.authenticate(req.username, req.password)
    if user is None:
        task_store.append_audit(
            owner_id=f"login-{login_identity}",
            role="anonymous",
            action="auth.login",
            resource_type="session",
            resource_id=None,
            request_id=getattr(request.state, "request_id", None),
            outcome="denied",
        )
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = portal_store.create_session(user["user_id"], settings.SESSION_TTL_SECONDS)
    task_store.append_audit(
        owner_id=user["user_id"],
        role=user["role"],
        action="auth.login",
        resource_type="session",
        resource_id=None,
        request_id=getattr(request.state, "request_id", None),
        outcome="accepted",
    )
    response = JSONResponse({"user": user})
    response.set_cookie(
        "analytics_session",
        token,
        max_age=settings.SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return response


@app.post("/auth/logout")
def logout(request: Request):
    principal = _request_principal(request)
    portal_store.revoke_session(request.cookies.get("analytics_session"))
    _record_audit(
        request,
        action="auth.logout",
        resource_type="session",
        resource_id=None,
        outcome="accepted",
    )
    response = JSONResponse({"subject": principal.subject, "logged_out": True})
    response.delete_cookie("analytics_session", path="/")
    return response


@app.get("/auth/me")
def current_identity(request: Request):
    principal = _request_principal(request)
    user = portal_store.get_user(principal.subject)
    payload = {
        "subject": principal.subject,
        "role": principal.role,
        "authentication_enabled": get_settings().API_AUTH_ENABLED,
        "web_login_enabled": get_settings().WEB_LOGIN_ENABLED,
    }
    if user:
        payload["user"] = user
    return payload


@app.get("/admin/users")
def list_portal_users(request: Request):
    _require_admin(request)
    users = portal_store.list_users()
    return {"users": users, "count": len(users)}


@app.post("/admin/users", status_code=201)
def create_portal_user(req: UserCreateRequest, request: Request):
    _require_admin(request)
    try:
        user = portal_store.create_user(
            req.username, req.password, req.display_name, req.role
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            raise HTTPException(status_code=409, detail="用户名已存在") from None
        raise
    _record_audit(
        request,
        action="user.create",
        resource_type="user",
        resource_id=user["user_id"],
        outcome="accepted",
        metadata={"role": user["role"]},
    )
    return user


@app.patch("/admin/users/{user_id}")
def update_portal_user(user_id: str, req: UserUpdateRequest, request: Request):
    principal = _require_admin(request)
    if user_id == principal.subject and req.enabled is False:
        raise HTTPException(status_code=409, detail="不能禁用当前登录账号")
    if user_id == principal.subject and req.role == "analyst":
        raise HTTPException(status_code=409, detail="不能降低当前登录账号的角色")
    try:
        user = portal_store.update_user(
            user_id,
            display_name=req.display_name,
            role=req.role,
            enabled=req.enabled,
            password=req.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    _record_audit(
        request,
        action="user.update",
        resource_type="user",
        resource_id=user_id,
        outcome="accepted",
    )
    return user


def _conversation_for_principal(conversation_id: str, request: Request) -> dict:
    conversation = portal_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    principal = _request_principal(request)
    if not principal.is_admin and conversation["owner_id"] != principal.subject:
        raise HTTPException(status_code=404, detail="对话不存在")
    pending_at = conversation.get("pending_clarification_at")
    pending_task_id = conversation.get("pending_task_id")
    clarification_ttl = getattr(get_settings(), "CLARIFICATION_TTL_SECONDS", 1800)
    if (
        pending_at is not None
        and pending_task_id
        and clarification_ttl >= 0
        and time() - float(pending_at) >= clarification_ttl
    ):
        try:
            task_store.request_cancel(pending_task_id)
        except KeyError:
            pass
        portal_store.close_task_link(pending_task_id)
        portal_store.clear_pending_clarification(conversation_id)
        conversation = portal_store.get_conversation(conversation_id) or conversation
    return conversation


@app.get("/conversations")
def list_conversations(request: Request, limit: int = 50):
    principal = _request_principal(request)
    conversations = portal_store.list_conversations(principal.subject, limit)
    return {"conversations": conversations, "count": len(conversations)}


@app.post("/conversations", status_code=201)
def create_conversation(req: ConversationCreateRequest, request: Request):
    principal = _request_principal(request)
    return portal_store.create_conversation(principal.subject, req.title)


@app.patch("/conversations/{conversation_id}")
def update_conversation(
    conversation_id: str, req: ConversationUpdateRequest, request: Request
):
    _conversation_for_principal(conversation_id, request)
    try:
        conversation = portal_store.update_conversation(conversation_id, req.title)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    _record_audit(
        request,
        action="conversation.update",
        resource_type="conversation",
        resource_id=conversation_id,
        outcome="accepted",
    )
    return conversation


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request):
    _conversation_for_principal(conversation_id, request)
    deleted = portal_store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="对话不存在")
    _record_audit(
        request,
        action="conversation.delete",
        resource_type="conversation",
        resource_id=conversation_id,
        outcome="accepted",
    )
    return {"conversation_id": conversation_id, "deleted": True}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, request: Request):
    conversation = _conversation_for_principal(conversation_id, request)
    principal = _request_principal(request)
    messages = portal_store.list_messages(conversation_id)
    return {
        **conversation,
        "messages": messages,
        "context_changes": portal_store.list_context_changes(conversation_id),
    }


@app.get("/metrics/catalog")
def get_metric_catalog(request: Request):
    """Expose the controlled, versioned metric definitions used by planning."""
    return metric_catalog_payload()


@app.get("/knowledge/catalog")
def get_knowledge_catalog(request: Request, data_source: str = "ai_analytics"):
    """Expose versioned, principal-filtered structural and business knowledge."""
    principal = _request_principal(request)
    schema_fields = get_schema_fields(data_source, principal.subject)
    if not schema_fields:
        raise HTTPException(status_code=404, detail="数据源不存在或无权访问")
    business = [
        item for item in business_knowledge_entries()
        if item["data_source"] == data_source
        and is_accessible(item, principal.subject)
    ]
    relationships = [
        item for item in SCHEMA_RELATIONSHIPS
        if item["data_source"] == data_source
        and is_accessible(item, principal.subject)
    ]
    return {
        "data_source": data_source,
        "schema_catalog_version": SCHEMA_CATALOG_VERSION,
        "schema_catalog_fingerprint": schema_catalog_fingerprint(),
        "schema_fields": schema_fields,
        "relationships": relationships,
        "business_knowledge": business,
    }


@app.post("/conversations/{conversation_id}/messages", status_code=202)
def send_conversation_message(
    conversation_id: str,
    req: ConversationMessageRequest,
    request: Request,
    response: Response,
):
    conversation = _conversation_for_principal(conversation_id, request)
    principal = _request_principal(request)
    _enforce_analysis_rate_limit(request, response)
    user_message = portal_store.add_message(
        conversation_id, "user", req.content
    )
    parsed_request = build_analysis_request(req.content)
    previous_request = (
        conversation.get("pending_clarification")
        or conversation.get("analysis_context")
    )
    effective_request, context_changes = inherit_analysis_request(
        previous_request, parsed_request, req.content
    )
    if effective_request.get("unsupported_conditions"):
        explanation = "当前无法按该口径执行：" + "；".join(
            effective_request["unsupported_conditions"]
        )
        portal_store.add_message(
            conversation_id,
            "system",
            explanation,
            metadata={
                "status": "unsupported",
                "analysis_request": effective_request,
                "context_changes": context_changes,
            },
        )
        pending_task_id = conversation.get("pending_task_id")
        if pending_task_id:
            task_store.finalize(
                pending_task_id,
                "failed",
                {"type": "task_failed", "reason": "unsupported_clarification"},
                error=explanation,
            )
            portal_store.close_task_link(pending_task_id)
            portal_store.clear_pending_clarification(conversation_id)
        return {
            "conversation_id": conversation_id,
            "message": user_message,
            "status": "unsupported",
            "detail": explanation,
            "analysis_request": effective_request,
            "context_changes": context_changes,
        }
    if effective_request.get("unresolved_ambiguities"):
        pending_task_id = conversation.get("pending_task_id")
        if not pending_task_id:
            pending_task_id = uuid4().hex
            task_store.create(
                pending_task_id, req.content, req.intent, owner_id=principal.subject
            )
            task_store.set_status(pending_task_id, "waiting_clarification")
            task_store.append_event(
                pending_task_id,
                {"type": "task_waiting_clarification"},
            )
            portal_store.link_task(
                pending_task_id, conversation_id, user_message["message_id"]
            )
        portal_store.save_analysis_context(
            conversation_id,
            effective_request,
            context_changes,
            source_message_id=user_message["message_id"],
            pending=True,
            pending_task_id=pending_task_id,
        )
        question = "执行前需要澄清：" + "；".join(
            effective_request["unresolved_ambiguities"]
        )
        portal_store.add_message(
            conversation_id,
            "system",
            question,
            metadata={
                "status": "waiting_clarification",
                "analysis_request": effective_request,
                "context_changes": context_changes,
            },
        )
        return {
            "conversation_id": conversation_id,
            "message": user_message,
            "status": "waiting_clarification",
            "task_id": pending_task_id,
            "status_url": f"/tasks/{pending_task_id}",
            "detail": question,
            "analysis_request": effective_request,
            "context_changes": context_changes,
        }

    pending_task_id = conversation.get("pending_task_id")
    portal_store.save_analysis_context(
        conversation_id,
        effective_request,
        context_changes,
        source_message_id=user_message["message_id"],
        pending=False,
    )
    execution_query = (
        CONFIRMED_REQUEST_PREFIX
        + json.dumps(effective_request, ensure_ascii=False, sort_keys=True)
        + f"\n当前用户表达：{req.content}"
    )
    try:
        if pending_task_id:
            resumed_task_id = task_runtime.resume_waiting(
                pending_task_id, execution_query, req.intent
            )
            receipt = SubmissionReceipt(resumed_task_id, replayed=False)
        else:
            receipt = task_runtime.submit_with_receipt(
                execution_query,
                req.intent,
                owner_id=principal.subject,
                conversation_id=conversation_id,
                user_message_id=user_message["message_id"],
            )
    except TaskQueueFull:
        portal_store.add_message(
            conversation_id, "system", "当前任务队列已满，请稍后重试。"
        )
        raise HTTPException(
            status_code=429,
            detail="分析任务队列已满，请稍后重试",
            headers={"Retry-After": "2"},
        ) from None
    return {
        "conversation_id": conversation_id,
        "message": user_message,
        "task_id": receipt.task_id,
        "status_url": f"/tasks/{receipt.task_id}",
        "events_url": f"/tasks/{receipt.task_id}/events",
        "status": "queued",
        "analysis_request": effective_request,
        "context_changes": context_changes,
    }


@app.get("/demo/scenarios")
def get_demo_scenarios():
    settings = get_settings()
    return {
        "enabled": settings.DEMO_MODE_ENABLED,
        "mode": "cached_local_dataset",
        "scenarios": list_demo_scenarios() if settings.DEMO_MODE_ENABLED else [],
    }


@app.post("/conversations/{conversation_id}/demo/{scenario_id}", status_code=201)
def run_demo_scenario(conversation_id: str, scenario_id: str, request: Request):
    """Persist a clearly-labelled deterministic demo result without invoking an LLM."""
    if not get_settings().DEMO_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="答辩演示模式未启用")
    _conversation_for_principal(conversation_id, request)
    principal = _request_principal(request)
    task_id = uuid4().hex
    try:
        scenario, result = build_demo_result(scenario_id, task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="演示场景不存在") from None

    task_store.create(task_id, scenario["query"], scenario["intent"], principal.subject)
    task_store.append_event(task_id, {"type": "task_queued", "demo_mode": True})
    if scenario.get("runtime_trace"):
        task_store.set_status(task_id, "running")
        task_store.append_event(task_id, {"type": "task_claimed", "worker_id": "demo-worker-a", "simulated": True})
        task_store.set_status(task_id, "queued")
        task_store.append_event(task_id, {"type": "task_requeued", "reason": "simulated_lease_expiry", "simulated": True})
        task_store.set_status(task_id, "running")
        task_store.append_event(task_id, {"type": "task_claimed", "worker_id": "demo-worker-b", "simulated": True})
    else:
        task_store.set_status(task_id, "running")
    task_store.append_event(task_id, {"type": "task_started", "demo_mode": True})
    user_message = portal_store.add_message(
        conversation_id,
        "user",
        scenario["query"],
        metadata={"demo_mode": True, "scenario_id": scenario_id},
    )
    portal_store.link_task(task_id, conversation_id, user_message["message_id"])
    task_store.finalize(
        task_id,
        "completed",
        {"type": "task_completed", "task_id": task_id, "demo_mode": True},
        result=result,
    )
    portal_store.complete_task(task_id, result, None)
    _record_audit(
        request,
        action="demo.run",
        resource_type="task",
        resource_id=task_id,
        outcome="completed",
        metadata={"scenario_id": scenario_id, "source": "fixed_local_dataset_snapshot"},
    )
    return {
        "conversation_id": conversation_id,
        "task_id": task_id,
        "status": "completed",
        "demo_mode": True,
        "scenario": {"id": scenario_id, "title": scenario["title"]},
        "status_url": f"/tasks/{task_id}",
    }


@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest, request: Request, response: Response):
    """
    对话式自然语言查询接口。

    接收用户的自然语言问题，调用 LangGraph 多智能体链路，
    返回 SQL / 查询结果 / 分析 / 预测 / 报告 / 图表 的完整结果。

    示例请求：
        POST /query
        {"query": "分析客户价值并预测流失风险"}
    """
    _enforce_analysis_rate_limit(request, response)
    request_id = request.state.request_id
    if not query_limiter.acquire():
        raise HTTPException(
            status_code=429,
            detail="当前分析任务已满，请稍后重试",
            headers={"Retry-After": "2"},
        )

    try:
        from agents.planner import run_query

        principal = _request_principal(request)
        with run_hooks(owner_id=principal.subject):
            state = run_query(req.query, requested_intent=req.intent)
        stored_result = {
            "intent": state.get("intent", ""),
            "report": state.get("report") or "",
            "charts": state.get("charts") or [],
        }
        stored_result["_owner_id"] = principal.subject
        result_store.put(request_id, stored_result)

        # 序列化 query_result（限制前100行避免响应过大）
        query_result = state.get("query_result")
        query_result_total_rows = (
            len(query_result) if query_result is not None else 0
        )
        query_result_truncated = query_result_total_rows > 100
        if query_result_truncated:
            query_result = query_result[:100]

        return QueryResponse(
            request_id=request_id,
            success=state.get("error") is None,
            intent=state.get("intent", ""),
            sql=state.get("sql"),
            query_result=query_result,
            query_result_total_rows=query_result_total_rows,
            query_result_truncated=query_result_truncated,
            governance_result=state.get("governance_result"),
            analysis_result=state.get("analysis_result"),
            prediction_result=state.get("prediction_result"),
            evidence=state.get("evidence"),
            report=state.get("report"),
            charts=state.get("charts"),
            error=state.get("error"),
            messages=state.get("messages"),
            execution_trace=state.get("execution_trace") or [],
        )

    except Exception as e:
        logger.exception("Query execution failed")
        return QueryResponse(
            request_id=request_id,
            success=False,
            intent="error",
            error="查询执行失败，请检查服务端日志",
            messages=[f"[ERROR] {type(e).__name__}"],
            execution_trace=[],
        )
    finally:
        query_limiter.release()


def _get_request_state(request_id: str, request: Request) -> dict:
    state = result_store.get(request_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="查询结果不存在或已过期，请重新提交 /query",
        )
    principal = _request_principal(request)
    owner_id = state.get("_owner_id")
    # Legacy ownerless results are never shared with analysts.
    if owner_id is None and not principal.is_admin:
        raise HTTPException(status_code=404, detail="查询结果不存在或已过期，请重新提交 /query")
    if owner_id is not None and not principal.is_admin and owner_id != principal.subject:
        raise HTTPException(status_code=404, detail="查询结果不存在或已过期，请重新提交 /query")
    state.pop("_owner_id", None)
    return state


@app.get("/charts/{request_id}")
def get_charts(request_id: str, request: Request):
    """获取指定请求生成的所有 ECharts 配置。"""
    state = _get_request_state(request_id, request)
    charts = state.get("charts") or []
    return {"request_id": request_id, "charts": charts, "count": len(charts)}


@app.get("/report/{request_id}")
def get_report(request_id: str, request: Request):
    """获取指定请求生成的经营洞察报告。"""
    state = _get_request_state(request_id, request)
    return {
        "request_id": request_id,
        "report": state.get("report") or "",
        "intent": state.get("intent", ""),
    }


# ---- Durable asynchronous task API ----

@app.post("/tasks", status_code=202)
def submit_task(
    req: TaskSubmission,
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    ),
):
    """Queue a durable analysis task and return immediately."""
    _enforce_analysis_rate_limit(request, response)
    principal = _request_principal(request)
    try:
        receipt = task_runtime.submit_with_receipt(
            req.query,
            req.intent,
            owner_id=principal.subject,
            idempotency_key=idempotency_key,
        )
    except IdempotencyConflict:
        _record_audit(
            request,
            action="task.submit",
            resource_type="task",
            resource_id=None,
            outcome="conflict",
            metadata={"idempotency_key_present": True},
        )
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key 已用于不同的任务内容",
        ) from None
    except TaskQueueFull:
        _record_audit(
            request,
            action="task.submit",
            resource_type="task",
            resource_id=None,
            outcome="queue_full",
        )
        raise HTTPException(
            status_code=429,
            detail="分析任务队列已满，请稍后重试",
            headers={"Retry-After": "2"},
        ) from None
    _record_audit(
        request,
        action="task.submit",
        resource_type="task",
        resource_id=receipt.task_id,
        outcome="replayed" if receipt.replayed else "accepted",
        metadata={"idempotency_key_present": bool(idempotency_key)},
    )
    task_status = (
        (task_store.get(receipt.task_id) or {}).get("status", "queued")
        if receipt.replayed
        else "queued"
    )
    return {
        "task_id": receipt.task_id,
        "status": task_status,
        "idempotency_replayed": receipt.replayed,
        "status_url": f"/tasks/{receipt.task_id}",
        "events_url": f"/tasks/{receipt.task_id}/events",
    }


@app.get("/tasks")
def list_tasks(request: Request, limit: int = 20, status: str | None = None):
    principal = _request_principal(request)
    owner_id = None if principal.is_admin else principal.subject
    tasks = task_store.list_recent(limit=limit, status=status, owner_id=owner_id)
    return {"tasks": tasks, "count": len(tasks)}


@app.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    return _task_for_principal(task_id, request)


@app.post("/tasks/{task_id}/retry", status_code=202)
def retry_task(task_id: str, request: Request, mode: str = "restart"):
    _task_for_principal(task_id, request)
    try:
        new_task_id = task_runtime.retry(task_id, mode=mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="分析任务不存在") from None
    except TaskQueueFull:
        raise HTTPException(
            status_code=429,
            detail="分析任务队列已满，请稍后重试",
            headers={"Retry-After": "2"},
        ) from None
    except UnsafeCheckpointResume as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "unsafe_checkpoint_resume",
                "message": str(exc),
                "safe_mode": "restart",
            },
        ) from None
    except ValueError:
        raise HTTPException(status_code=422, detail="mode 必须是 restart 或 checkpoint") from None
    except RuntimeError:
        raise HTTPException(status_code=409, detail="任务仍在执行，不能重复提交") from None
    _record_audit(
        request,
        action="task.retry",
        resource_type="task",
        resource_id=new_task_id,
        outcome="accepted",
        metadata={"retry_of": task_id, "mode": mode},
    )
    return {
        "task_id": new_task_id,
        "retry_of": task_id,
        "retry_mode": mode,
        "duplicate_execution_warning": True,
        "status": "queued",
        "status_url": f"/tasks/{new_task_id}",
        "events_url": f"/tasks/{new_task_id}/events",
    }


@app.delete("/tasks/{task_id}", status_code=202)
def cancel_task(task_id: str, request: Request):
    _task_for_principal(task_id, request)
    try:
        status = task_runtime.cancel(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="分析任务不存在") from None
    if status == "cancelled":
        portal_store.close_task_link(task_id)
        portal_store.clear_pending_for_task(task_id)
    _record_audit(
        request,
        action="task.cancel",
        resource_type="task",
        resource_id=task_id,
        outcome=status,
    )
    return {"task_id": task_id, "status": status}


@app.get("/tasks/{task_id}/checkpoint")
def get_task_checkpoint(task_id: str, request: Request):
    _task_for_principal(task_id, request)
    checkpoint = task_store.latest_checkpoint(task_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="任务尚未产生检查点")
    return checkpoint


@app.get("/tasks/{task_id}/events")
async def stream_task_events(task_id: str, request: Request, after: int = 0):
    """Stream durable task events with Server-Sent Events (SSE)."""
    _task_for_principal(task_id, request)

    try:
        last_sequence = max(after, int(request.headers.get("Last-Event-ID", "0")))
    except ValueError:
        last_sequence = after
    poll_seconds = get_settings().TASK_EVENT_POLL_SECONDS

    async def generate():
        nonlocal last_sequence
        while True:
            events = task_store.events_after(task_id, last_sequence)
            for event in events:
                last_sequence = event["sequence"]
                event_type = event.get("type", "message")
                payload = json.dumps(event, ensure_ascii=False, default=str)
                yield f"id: {last_sequence}\nevent: {event_type}\ndata: {payload}\n\n"

            task = task_store.get(task_id)
            if task is None or task["status"] in TERMINAL_STATUSES:
                break
            if await request.is_disconnected():
                break
            await asyncio.sleep(poll_seconds)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/operations/summary")
def operations_summary(request: Request):
    """Return local task runtime metrics without exposing prompts or results."""
    principal = _request_principal(request)
    owner_id = None if principal.is_admin else principal.subject
    workers = task_store.list_workers()
    return {
        **task_store.summary(owner_id=owner_id),
        "usage": task_store.usage_summary(owner_id=owner_id),
        "worker_capacity": get_settings().MAX_CONCURRENT_QUERIES,
        "queue_capacity": get_settings().MAX_QUEUED_TASKS,
        "active_workers": len(workers),
        "execution_mode": get_settings().TASK_EXECUTION_MODE,
        "audit_event_count": task_store.audit_count(owner_id=owner_id),
        "rate_limit_rejections": task_store.audit_outcome_count(
            "rate_limited", owner_id=owner_id
        ),
    }


@app.get("/operations/workers")
def operations_workers(request: Request):
    _require_admin(request)
    workers = task_store.list_workers()
    return {"workers": workers, "count": len(workers)}


@app.get("/operations/audit")
def operations_audit(
    request: Request,
    limit: int = 50,
    after: int = 0,
    owner_id: str | None = None,
):
    """Return durable mutation audit records to administrators only."""
    _require_admin(request)
    events = task_store.list_audit(limit=limit, after=after, owner_id=owner_id)
    return {
        "events": events,
        "count": len(events),
        "next_after": events[-1]["sequence"] if events else after,
    }


@app.get("/operations/rate-limits")
def operations_rate_limits(request: Request, limit: int = 200):
    """Return active shared rate-limit windows to administrators only."""
    _require_admin(request)
    settings = get_settings()
    window_seconds = settings.API_RATE_LIMIT_WINDOW_SECONDS
    if settings.API_RATE_LIMIT_REQUESTS < 0 or window_seconds < 1:
        raise HTTPException(status_code=503, detail="API 限流配置无效")
    window_start = int(time() // window_seconds) * window_seconds
    windows = task_store.list_rate_limits(
        current_window_start=window_start, limit=limit
    )
    for item in windows:
        item["limit"] = settings.API_RATE_LIMIT_REQUESTS
        item["remaining"] = max(
            0, settings.API_RATE_LIMIT_REQUESTS - item["request_count"]
        )
        item["reset_at"] = item["window_start"] + window_seconds
    return {"windows": windows, "count": len(windows)}


@app.get("/operations/preflight", response_model=PreflightReport)
def operations_preflight(request: Request):
    """Run read-only checks required for a reliable defense demonstration."""
    _require_admin(request)
    return run_preflight(task_store, get_settings())


@app.get("/data/catalog")
def data_catalog(request: Request):
    tables = load_catalog(get_settings())
    return {"tables": tables, "count": len(tables)}


@app.get("/data/tables/{table_name}/schema")
def data_table_schema(table_name: str, request: Request):
    try:
        return load_table_schema(table_name, get_settings())
    except KeyError:
        raise HTTPException(status_code=404, detail="数据表不存在或不允许访问") from None


@app.get("/data/tables/{table_name}/preview")
def data_table_preview(table_name: str, request: Request, limit: int = 20):
    try:
        return load_preview(table_name, get_settings(), limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="数据表不存在或不允许访问") from None


@app.get("/operations/metrics", response_class=PlainTextResponse)
def prometheus_metrics(request: Request):
    """Expose tenant-scoped task metrics in Prometheus text format."""
    principal = _request_principal(request)
    owner_id = None if principal.is_admin else principal.subject
    summary = task_store.summary(owner_id=owner_id)
    lines = [
        "# HELP ai_analytics_tasks_total Number of durable tasks by status.",
        "# TYPE ai_analytics_tasks_total gauge",
    ]
    for status, count in sorted(summary["status_counts"].items()):
        safe_status = "".join(
            character for character in status if character.isalnum() or character == "_"
        )
        lines.append(
            f'ai_analytics_tasks_total{{status="{safe_status}"}} {int(count)}'
        )
    lines.extend([
        "# HELP ai_analytics_task_duration_seconds Average completed task duration.",
        "# TYPE ai_analytics_task_duration_seconds gauge",
        f'ai_analytics_task_duration_seconds {summary["completed_avg_seconds"]}',
        "# HELP ai_analytics_task_events_total Persisted workflow events.",
        "# TYPE ai_analytics_task_events_total gauge",
        f'ai_analytics_task_events_total {summary["event_count"]}',
        "# HELP ai_analytics_task_checkpoints_total Persisted workflow checkpoints.",
        "# TYPE ai_analytics_task_checkpoints_total gauge",
        f'ai_analytics_task_checkpoints_total {summary["checkpoint_count"]}',
        "# HELP ai_analytics_worker_capacity Configured background worker capacity.",
        "# TYPE ai_analytics_worker_capacity gauge",
        f'ai_analytics_worker_capacity {get_settings().MAX_CONCURRENT_QUERIES}',
        "# HELP ai_analytics_queue_capacity Configured waiting-task capacity.",
        "# TYPE ai_analytics_queue_capacity gauge",
        f'ai_analytics_queue_capacity {get_settings().MAX_QUEUED_TASKS}',
        "# HELP ai_analytics_workers_active Workers with a recent heartbeat.",
        "# TYPE ai_analytics_workers_active gauge",
        f"ai_analytics_workers_active {len(task_store.list_workers())}",
        "# HELP ai_analytics_audit_events_total Persisted mutation audit records.",
        "# TYPE ai_analytics_audit_events_total gauge",
        f"ai_analytics_audit_events_total {task_store.audit_count(owner_id=owner_id)}",
        "# HELP ai_analytics_rate_limit_rejections_retained Rejections in retained audit records.",
        "# TYPE ai_analytics_rate_limit_rejections_retained gauge",
        "ai_analytics_rate_limit_rejections_retained "
        f'{task_store.audit_outcome_count("rate_limited", owner_id=owner_id)}',
    ])
    return "\n".join(lines) + "\n"


@app.get("/")
def serve_frontend():
    """Serve the built Vue portal, with the legacy dashboard as a safe fallback."""
    frontend_path = FRONTEND_DIST_DIR / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    legacy_path = FRONTEND_DIR / "legacy.html"
    if legacy_path.exists():
        return FileResponse(legacy_path)
    return JSONResponse(
        {"message": "前端尚未构建，请在 frontend 目录运行 npm run build"},
        status_code=404,
    )


@app.get("/legacy")
def serve_legacy_frontend():
    """Keep the former single-page dashboard available for demonstrations."""
    legacy_path = FRONTEND_DIR / "legacy.html"
    if legacy_path.exists():
        return FileResponse(legacy_path)
    raise HTTPException(status_code=404, detail="旧版前端不存在")


# ---- 启动入口 ----

def main():
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
