from argparse import Namespace

import pytest

from evaluation.experiments.architecture_comparison_v2 import (
    CONFIRMATION, _failure_category, build_protocol,
    validate_execution_authorization,
)
from evaluation.experiments.stage5_deterministic_regression import run
from evaluation.framework.stage5 import (
    _hash, audit_catalog, summarize_records, validate_sample_record,
)


def _args(**updates):
    values = {
        "execute": False, "confirm": "", "model": "fixed-model",
        "max_samples": 5, "max_total_calls": 0, "max_total_tokens": 0,
        "max_cost_usd": 0.0, "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }
    values.update(updates)
    return Namespace(**values)


def test_versioned_catalog_audits_179_samples_and_disjoint_splits():
    audit = audit_catalog()
    assert audit["status"] == "passed"
    assert audit["total_samples"] == 179
    assert audit["dataset_count"] == 7
    assert not audit["issues"]


def test_dataset_hash_is_independent_of_platform_line_endings(tmp_path):
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'[\n  {"id": 1}\n]\n')
    crlf.write_bytes(b'[\r\n  {"id": 1}\r\n]\r\n')
    assert _hash(lf) == _hash(crlf)


def test_deterministic_stage5_regression_has_no_model_usage():
    result = run()
    assert result["success"] is True
    assert len(result["records"]) == 31
    assert all(item["llm_calls"] == 0 and item["total_tokens"] == 0 for item in result["records"])
    assert result["metrics"]["multiturn"]["task_completion_rate"]["value"] == 1.0


def test_metric_denominators_exclude_unlabelled_samples():
    metrics = summarize_records([
        {"duration_ms": 10, "answer_correct": True},
        {"duration_ms": 20, "answer_correct": False},
        {"duration_ms": 30},
    ])
    assert metrics["query"]["business_answer_accuracy"] == {"value": 0.5, "n": 2}
    assert metrics["report"]["citation_correct_rate"] == {"value": None, "n": 0}
    assert metrics["engineering"]["latency_p95_ms"] == 30.0


def test_sample_record_requires_failure_category_and_resource_fields():
    errors = validate_sample_record({"success": False})
    assert "failed records require failure_category" in errors
    assert any("total_tokens" in item for item in errors)


def test_architecture_comparison_defaults_to_dry_run_and_equal_permissions():
    args = _args()
    validate_execution_authorization(args)
    protocol = build_protocol(args)
    assert set(protocol["architectures"]) == {
        "direct_sql", "single_agent_tools", "controlled_workflow"
    }
    assert protocol["historical_results_reused"] is False
    assert "tables" in protocol["shared_permissions"]


def test_paid_comparison_refuses_missing_confirmation_or_caps():
    with pytest.raises(ValueError, match="requires --confirm"):
        validate_execution_authorization(_args(execute=True))
    with pytest.raises(ValueError, match="positive caps"):
        validate_execution_authorization(_args(execute=True, confirm=CONFIRMATION))
    validate_execution_authorization(_args(
        execute=True, confirm=CONFIRMATION, max_samples=2,
        max_total_calls=20, max_total_tokens=20000, max_cost_usd=1.0,
        input_cost_per_million=0.2, output_cost_per_million=0.4,
    ))


def test_failed_labelled_query_is_a_failure_not_a_missing_label():
    detail = {"success": False, "execution_success": False, "result_correct": None}
    assert _failure_category(detail) == "execution_failure"
    assert bool(detail.get("result_correct")) is False
