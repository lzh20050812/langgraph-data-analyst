"""
FastAPI 应用层 —— 对外暴露对话式查询接口 + 图表数据接口。

接口：
- POST /query    对话式自然语言查询（支持 sql_query / analysis / prediction / mixed 全链路）
- GET  /charts   获取最近一次查询生成的图表数据列表
- GET  /health   健康检查
- GET  /         静态前端看板 (frontend/index.html)

启动方式：
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    或: python -m api.main
"""

import sys
import traceback
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="AI Data Analyst — 智能运营分析平台",
    description="基于 LangGraph 的多智能体企业运营分析 API",
    version="1.0.0",
)

# ---- 请求模型 ----

class QueryRequest(BaseModel):
    query: str
    intent: Optional[str] = None  # 可选：手动指定意图，留空则自动解析

class QueryResponse(BaseModel):
    success: bool
    intent: str
    sql: Optional[str] = None
    query_result: Optional[list] = None
    governance_result: Optional[dict] = None
    analysis_result: Optional[dict] = None
    prediction_result: Optional[dict] = None
    report: Optional[str] = None
    charts: Optional[list] = None
    error: Optional[str] = None
    messages: Optional[list] = None


# ---- 全局缓存（最近一次查询的结果） ----

_latest_state: dict = {}
_latest_charts: list = []


# ---- 接口 ----

@app.get("/health")
def health_check():
    """健康检查：验证 MySQL 和 LLM 是否可用。"""
    health = {
        "status": "ok",
        "mysql": False,
        "llm": False,
    }

    # MySQL
    try:
        from storage.mysql.client import check_connection
        health["mysql"] = check_connection()
    except Exception:
        pass

    # LLM
    try:
        from config.settings import get_settings
        settings = get_settings()
        health["llm"] = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)
    except Exception:
        pass

    if not health["mysql"] or not health["llm"]:
        health["status"] = "degraded"

    return health


@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    """
    对话式自然语言查询接口。

    接收用户的自然语言问题，调用 LangGraph 多智能体链路，
    返回 SQL / 查询结果 / 分析 / 预测 / 报告 / 图表 的完整结果。

    示例请求：
        POST /query
        {"query": "分析客户价值并预测流失风险"}
    """
    global _latest_state, _latest_charts

    try:
        from agents.planner import run_query

        state = run_query(req.query)

        # 缓存最新结果
        _latest_state = dict(state)
        _latest_charts = state.get("charts", [])

        # 序列化 query_result（限制前100行避免响应过大）
        query_result = state.get("query_result")
        if query_result and len(query_result) > 100:
            query_result = query_result[:100]

        return QueryResponse(
            success=state.get("error") is None,
            intent=state.get("intent", ""),
            sql=state.get("sql"),
            query_result=query_result,
            governance_result=state.get("governance_result"),
            analysis_result=state.get("analysis_result"),
            prediction_result=state.get("prediction_result"),
            report=state.get("report"),
            charts=state.get("charts"),
            error=state.get("error"),
            messages=state.get("messages"),
        )

    except Exception as e:
        return QueryResponse(
            success=False,
            intent="error",
            error=f"{e}\n{traceback.format_exc()}",
            messages=[f"[ERROR] {e}"],
        )


@app.get("/charts")
def get_charts():
    """获取最近一次查询生成的所有图表配置（ECharts option JSON）。"""
    return {"charts": _latest_charts, "count": len(_latest_charts)}


@app.get("/report")
def get_report():
    """获取最近一次查询生成的经营洞察报告。"""
    report = _latest_state.get("report", "")
    intent = _latest_state.get("intent", "")
    return {"report": report, "intent": intent}


@app.get("/")
def serve_frontend():
    """前端看板入口。"""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return JSONResponse(
        {"message": "前端文件不存在，请访问 /docs 查看 API 文档"},
        status_code=404,
    )


# ---- 启动入口 ----

def main():
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
