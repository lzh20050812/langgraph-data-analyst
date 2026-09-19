from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "legacy.html"
).read_text(encoding="utf-8")


def test_frontend_renders_query_rows_and_supports_safe_csv_export():
    assert "function renderQueryTable" in FRONTEND
    assert "function downloadQueryCsv" in FRONTEND
    assert "query_result_total_rows" in FRONTEND
    assert "Prevent spreadsheet formula execution" in FRONTEND


def test_frontend_exposes_validation_and_test_model_comparison():
    assert "function renderModelComparison" in FRONTEND
    assert "candidate_metrics" in FRONTEND
    assert "test_candidate_metrics" in FRONTEND
    assert "独立测试结果" in FRONTEND


def test_frontend_connects_to_local_api_when_opened_from_disk():
    assert "window.location.protocol === 'file:'" in FRONTEND
    assert "http://127.0.0.1:8000" in FRONTEND
    assert "fetch(`${API_BASE}/health`)" in FRONTEND


def test_frontend_streams_progress_and_supports_task_lifecycle():
    assert "response.body.getReader()" in FRONTEND
    assert "function cancelActiveTask" in FRONTEND
    assert "function showRecentTasks" in FRONTEND
    assert "X-API-Key" in FRONTEND
    assert "Idempotency-Key" in FRONTEND
    assert "pendingAnalysisSubmission" in FRONTEND
    assert "Retry-After" in FRONTEND


def test_frontend_exposes_live_operations_dashboard():
    assert "function showOperationsDashboard" in FRONTEND
    assert "function renderOperationsDashboard" in FRONTEND
    assert "/operations/workers" in FRONTEND
    assert "/operations/audit?limit=8" in FRONTEND
    assert "/operations/rate-limits?limit=8" in FRONTEND
    assert "/operations/preflight" in FRONTEND
    assert "多智能体执行链路" in FRONTEND
    assert "答辩预检清单" in FRONTEND


def test_frontend_degrades_gracefully_without_echarts_cdn():
    assert "if (!window.echarts)" in FRONTEND
    assert "查询表格与报告数据仍可正常查看" in FRONTEND
