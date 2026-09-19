import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "evaluation" / "final_results"


def _load(name: str) -> dict:
    return json.loads((FINAL_DIR / name).read_text(encoding="utf-8"))


def test_final_metrics_are_frozen_and_internally_consistent():
    metrics = _load("final_metrics.json")

    assert metrics["status"] == "final"
    assert metrics["frozen_at"] == "2026-08-12"
    assert metrics["regression_tests"]["material_pack_passed"] == 32
    assert metrics["regression_tests"]["current_passed"] == 55
    assert metrics["text2sql"]["scorable_samples"] == 49
    assert metrics["text2sql"]["rag_content_execution_accuracy"] == 0.877551
    assert metrics["churn_prediction"]["xgboost_auc"] == 0.677656
    assert metrics["revenue_forecast"]["prophet_mape_pct"] == 15.6457
    assert metrics["revenue_forecast"]["last_value_mape_pct"] == 14.3602
    assert metrics["business_semantics_ablation"]["mcnemar_p_two_sided"] == 0.0078
    assert metrics["system"]["throughput_rps_at_4_workers"] == 1.3922


def test_readme_uses_the_frozen_headline_metrics():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected_claims = [
        "BGE Field R@5 84.58%; character TF-IDF 90.00%",
        "RAG execution accuracy 87.76%; full-schema 79.59%",
        "Enabled 100%; disabled 46.67%",
        "XGBoost AUC 0.6777; logistic AUC 0.6453",
        "Prophet MAPE 15.65%; last-value baseline 14.36%",
        "R@1 80%; R@3 90%; MRR@3 0.85",
        "Four-worker throughput: 1.392 req/s",
    ]

    for claim in expected_claims:
        assert claim in readme

    assert "evaluation/final_results/final_metrics.json" in readme
    assert "55 tests passed" in readme


def test_local_source_artifacts_match_the_frozen_manifest_when_available():
    manifest = _load("source_manifest.json")

    for source in manifest["sources"].values():
        path = ROOT / source["path"]
        if not path.exists():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == source["sha256"], source["path"]
