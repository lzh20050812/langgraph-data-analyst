from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def test_vue_portal_has_authenticated_multi_page_routes():
    router = (FRONTEND / "src" / "router" / "index.ts").read_text(encoding="utf-8")
    for route in ("/login", "chat", "data", "tasks", "operations", "audit", "users"):
        assert route in router
    assert "meta: { public: true }" in router
    assert "admin: true" in router


def test_browser_auth_uses_http_only_cookie_contract():
    client = (FRONTEND / "src" / "api" / "client.ts").read_text(encoding="utf-8")
    store = (FRONTEND / "src" / "stores" / "auth.ts").read_text(encoding="utf-8")
    combined = client + store
    assert "credentials: 'include'" in combined
    assert "/auth/login" in combined
    assert "/auth/logout" in combined
    assert "localStorage" not in combined
    assert "sessionStorage" not in combined


def test_frontend_build_and_legacy_fallback_are_present():
    assert (FRONTEND / "dist" / "index.html").is_file()
    assert (FRONTEND / "legacy.html").is_file()
    package = (FRONTEND / "package.json").read_text(encoding="utf-8")
    assert '"vue"' in package
    assert '"build"' in package


def test_frontend_keeps_bundle_budget_and_component_level_imports():
    main = (FRONTEND / "src" / "main.ts").read_text(encoding="utf-8")
    assert "import ElementPlus from" not in main
    assert "element-plus/dist/index.css" not in main
    assets = FRONTEND / "dist" / "assets"
    main_bundles = list(assets.glob("index-*.js"))
    chart_bundles = list(assets.glob("chartRuntime-*.js"))
    assert main_bundles and max(path.stat().st_size for path in main_bundles) < 500_000
    assert chart_bundles and max(path.stat().st_size for path in chart_bundles) < 650_000
    assert max(path.stat().st_size for path in assets.glob("*.css")) < 200_000


def test_long_lived_frontend_resources_are_cleaned_up():
    chat = (FRONTEND / "src" / "views" / "ChatView.vue").read_text(encoding="utf-8")
    operations = (FRONTEND / "src" / "views" / "OperationsView.vue").read_text(encoding="utf-8")
    assert "onUnmounted" in chat
    assert "activeStreams" in chat
    assert "onUnmounted" in operations
    assert "clearInterval" in operations


def test_user_management_supports_safe_editing():
    users = (FRONTEND / "src" / "views" / "UsersView.vue").read_text(encoding="utf-8")
    assert "编辑登录用户" in users
    assert "重置密码" in users
    assert "editing?.user_id===auth.user?.user_id" in users


def test_structured_results_restore_charts_in_chat_and_task_details():
    chat = (FRONTEND / "src" / "views" / "ChatView.vue").read_text(encoding="utf-8")
    tasks = (FRONTEND / "src" / "views" / "TasksView.vue").read_text(encoding="utf-8")
    visuals = (FRONTEND / "src" / "components" / "ResultVisuals.vue").read_text(encoding="utf-8")
    chart = (FRONTEND / "src" / "components" / "ChartPanel.vue").read_text(encoding="utf-8")
    assert "hydrateTaskResults" in chat
    assert "<ResultVisuals" in chat
    assert "<ResultVisuals" in tasks
    assert "<ChartPanel" in visuals
    assert "query_result" in visuals
    assert "downloadCsv" in visuals
    assert "import('../lib/chartRuntime')" in chart
    assert "ResizeObserver" in chart


def test_defense_demo_and_conversation_management_are_explicit():
    chat = (FRONTEND / "src" / "views" / "ChatView.vue").read_text(encoding="utf-8")
    report = (FRONTEND / "src" / "components" / "StructuredReport.vue").read_text(encoding="utf-8")
    visuals = (FRONTEND / "src" / "components" / "ResultVisuals.vue").read_text(encoding="utf-8")
    assert "/demo/scenarios" in chat
    assert "固定数据集缓存 · 不调用 LLM" in chat
    assert "renameConversation" in chat and "deleteConversation" in chat
    assert "StructuredReport" in chat and "StructuredReport" in (FRONTEND / "src" / "views" / "TasksView.vue").read_text(encoding="utf-8")
    assert "v-html" not in report
    assert "缓存演示结果" in visuals and "证据来源" in visuals


def test_chat_displays_effective_analysis_context_and_clarification_state():
    chat = (FRONTEND / "src" / "views" / "ChatView.vue").read_text(encoding="utf-8")
    types = (FRONTEND / "src" / "types" / "index.ts").read_text(encoding="utf-8")
    assert "analysis_context" in chat and "pending_clarification" in chat
    assert "本次分析条件" in chat and "待澄清条件" in chat
    assert "修改条件" in chat and "prepareConditionEdit('new')" in chat
    assert "waiting_clarification" in chat and "unsupported" in chat
    assert "interface AnalysisRequest" in types


def test_task_recovery_controls_and_result_modes_are_explicit():
    tasks = (FRONTEND / "src" / "views" / "TasksView.vue").read_text(encoding="utf-8")
    visuals = (FRONTEND / "src" / "components" / "ResultVisuals.vue").read_text(encoding="utf-8")
    assert "确认取消任务" in tasks
    assert "确认从头重试" in tasks
    assert "route.query.task" in tasks
    assert "缓存演示结果" in visuals
    assert "实时 Agent 分析" in visuals
    assert "故障恢复时间线" in visuals
