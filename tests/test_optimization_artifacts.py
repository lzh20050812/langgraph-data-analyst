import hashlib
import inspect
import json
from pathlib import Path

from agents.schema_retrieval import fuse_schema_candidates


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "evaluation/optimization_results/schema_hybrid_20260825"
PREDICTION_DIR = (
    ROOT / "evaluation/optimization_results/prediction_selection_20260825"
)


def test_hybrid_schema_artifact_uses_disjoint_validation_and_test_sets():
    metrics = json.loads((SCHEMA_DIR / "metrics.json").read_text(encoding="utf-8"))
    validation = set(metrics["split"]["validation_ids"])
    test = set(metrics["split"]["test_ids"])

    assert validation.isdisjoint(test)
    assert len(validation | test) == 40
    assert metrics["test_summaries"]["hybrid"]["field_recall_at_5"] >= (
        metrics["test_summaries"]["lexical"]["field_recall_at_5"]
    )
    assert (SCHEMA_DIR / "schema_hybrid_heldout.png").stat().st_size > 10_000


def test_runtime_hybrid_defaults_match_validation_selected_parameters():
    metrics = json.loads((SCHEMA_DIR / "metrics.json").read_text(encoding="utf-8"))
    selected = metrics["parameter_selection"]["selected"]
    signature = inspect.signature(fuse_schema_candidates)

    assert signature.parameters["vector_weight"].default == selected["vector_weight"]
    assert signature.parameters["lexical_weight"].default == selected["lexical_weight"]
    assert signature.parameters["rrf_k"].default == selected["rrf_k"]
    source = ROOT / "agents/schema_retrieval.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        metrics["runtime_retriever_sha256"]
    )


def test_prediction_selection_keeps_final_holdout_out_of_model_choice():
    metrics = json.loads(
        (PREDICTION_DIR / "metrics.json").read_text(encoding="utf-8")
    )
    validation_best = min(
        metrics["candidate_metrics"],
        key=lambda name: metrics["candidate_metrics"][name][
            "validation_mape_mean"
        ],
    )
    test_best = min(
        metrics["test_candidate_metrics"],
        key=lambda name: metrics["test_candidate_metrics"][name]["mape_pct"],
    )

    assert metrics["selection_set"] == "rolling_validation"
    assert metrics["test_set"] == "final_holdout"
    assert metrics["selected_model"] == validation_best == "Prophet"
    assert test_best == "LastValue"
    assert len(metrics["validation_folds"]) == 3
    assert (PREDICTION_DIR / "forecast_model_selection.png").stat().st_size > 10_000
