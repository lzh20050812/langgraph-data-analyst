from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_defense_launcher_opens_operations_view():
    launcher = (ROOT / "答辩演示.bat").read_text(encoding="utf-8")
    startup = (ROOT / "scripts" / "start_local.ps1").read_text(encoding="utf-8")
    frontend = (ROOT / "frontend" / "src" / "router" / "index.ts").read_text(encoding="utf-8")

    assert "-Operations" in launcher
    assert "[switch]$Operations" in startup
    assert "#/operations" in startup
    assert "path: 'operations'" in frontend


def test_defense_guide_covers_demo_and_fallback_paths():
    guide = (ROOT / "docs" / "DEFENSE_DEMO.md").read_text(encoding="utf-8")
    assert "6 分钟演示流程" in guide
    assert "SQLGlot AST" in guide
    assert "现场兜底" in guide


def test_demo_catalog_covers_three_required_storylines():
    from api.demo_scenarios import list_demo_scenarios

    kinds = {item["kind"] for item in list_demo_scenarios()}
    assert {"correct_analysis", "multiturn_clarification", "failure_recovery"} <= kinds


def test_release_verifier_covers_backend_evaluation_and_frontend():
    script = (ROOT / "scripts" / "verify_release.ps1").read_text(encoding="utf-8")
    assert "-m pytest -q" in script
    assert "stage5_deterministic_regression --check" in script
    assert "npm run typecheck" in script
    assert "npm run build" in script
