# Reproducible Benchmarks

This document describes the evaluation boundary for AI Data Analyst. All metrics are generated from fixed local test sets and are intended for regression tracking, architecture comparison, and failure analysis. They are not claims about unrestricted open-domain performance.

## Benchmark matrix

| Area | Entry point | Primary metrics |
|---|---|---|
| Data governance | `evaluation/experiments/governance_experiment.py` | missing-value repair, duplicate repair, record retention, repeat runtime |
| Schema retrieval | `evaluation/experiments/rag_retrieval_formal.py` | Field/Table Recall@K, MRR, paired significance |
| Held-out hybrid retrieval | `evaluation/experiments/hybrid_schema_evaluation.py` | validation-selected RRF parameters, held-out Field Recall@K, bootstrap delta |
| Text2SQL | `evaluation/experiments/text2sql_ablation_formal.py` | execution accuracy, first-pass accuracy, retry recovery, McNemar test |
| Multi-agent orchestration | `evaluation/experiments/run_formal_comparison.py` | task completion score, SQL accuracy, report quality, route efficiency |
| Analytics and prediction | `evaluation/experiments/model_experiment_formal.py` | silhouette, ARI, ROC-AUC, MAPE, baseline deltas |
| Forecast model selection | `evaluation/experiments/prediction_selection_evaluation.py` | rolling-validation MAPE, independent holdout MAPE, selection stability |
| Runtime and API | `evaluation/experiments/system_experiment_formal.py` | contract compliance, latency, throughput, concurrency success rate |
| Security and scope | `evaluation/experiments/robustness_experiment.py` | OOD fail-closed rate, prompt-injection isolation, database invariants |
| Analysis memory | `evaluation/experiments/memory_reuse_experiment.py` | Recall@K, MRR, prompt integration, random baseline |
| Versioned deterministic regression | `evaluation/experiments/stage5_deterministic_regression.py` | dataset audit, semantic accuracy, transfer retrieval, relationship coverage, multi-turn correctness |
| Same-model architecture comparison | `evaluation/experiments/architecture_comparison_v2.py` | business-answer accuracy, execution success, P50/P95, calls, tokens, estimated cost |

## Test data

- `evaluation/data/text2sql/`: fixed Text2SQL questions and audited references.
- `evaluation/data/rag_schema/`: Schema retrieval queries and field/table labels.
- `evaluation/data/multi_agent/`: multi-intent tasks for orchestration comparison.
- `evaluation/data/benchmark_catalog_v2026_09.json`: versioned 179-sample catalog with hashes and usage boundaries.
- `evaluation/data/multiturn/`: context inheritance, replacement, clearing, and clarification cases.
- `evaluation/data/failure_cases/`: stable query/report failure categories linked to executable regressions.

See [`EVALUATION_PROTOCOL_V2.md`](EVALUATION_PROTOCOL_V2.md) for split policy,
independent metric denominators, blinded human review, per-sample artifacts, and
the explicit call/token/cost gate for paid comparisons.

## Reproduction principles

1. Freeze the code, test-set snapshot, dependency versions, model name, and random seed.
2. Compare paired systems on the same samples and preserve per-sample outcomes.
3. Report strong deterministic or statistical baselines, including negative results.
4. Separate syntax validity, execution success, and business-semantic correctness.
5. Keep LLM-judged metrics secondary to executable and deterministic evidence.
6. Do not commit API keys, runtime logs, local model caches, or large raw outputs.

## Current validated results

The frozen machine-readable baseline is [`evaluation/final_results/final_metrics.json`](../evaluation/final_results/final_metrics.json), with source paths and SHA-256 hashes recorded in [`source_manifest.json`](../evaluation/final_results/source_manifest.json). The tracked `data/processed/eval_*.json` files predate this formal run and are not the thesis baseline.

- Data governance repaired 1,176 missing cells and 604 duplicate issues while retaining 8,000 unique customers.
- Text2SQL content execution accuracy reached 87.76% on 49 scorable questions.
- The deterministic business-semantics layer improved a 15-case high-risk subset from 46.67% to 100% (`p=0.0078`).
- XGBoost reached ROC-AUC 0.6777 versus 0.6453 for logistic regression; the confidence interval for the delta crossed zero.
- Prophet reached MAPE 15.65% and did not beat the last-value baseline at 14.36%.
- OOD requests failed closed in 6/6 cases and prompt-injection attempts were isolated in 4/4 cases.
- API success rate remained 100% at 1, 2, and 4 concurrent workers.

The repository intentionally keeps the benchmark harness and fixed test sets under version control while excluding bulky local run artifacts. This keeps clones small and makes new results independently reproducible.

## Post-freeze optimization results

The 2026-08-25 optimization artifacts are intentionally separate from the frozen
2026-08-12 baseline:

- `evaluation/optimization_results/schema_hybrid_20260825/` records a 12-question
  validation split and 28-question held-out split. Hybrid Field Recall@5 was
  86.31% versus 85.71% for the validation-selected lexical reference; the paired
  bootstrap interval crossed zero.
- `evaluation/optimization_results/prediction_selection_20260825/` records three
  rolling validation folds and a final six-month holdout. Prophet won validation
  selection at 9.12% mean MAPE but reached 15.65% on the holdout, where the
  LastValue audit baseline reached 14.36%. The holdout is reported, not reused for
  model selection.

The 2026-09-18 Stage 5 deterministic artifact is also post-freeze. It audits 179
cataloged samples and runs 31 no-LLM regressions. It must not be interpreted as a
current LLM architecture comparison. The older 2026-08-11 three-way experiment
is retained as historical evidence, but its single-LLM lower bound lacked private
Schema context and is not reused as a fair V2 comparison.

The authorized V2 pilot at
`evaluation/results/architecture_comparison_v2/20260918_160657/` used
`deepseek-v4-flash` on five known-template questions per architecture. Strict
answer equivalence was 0/5 for direct SQL, 0/5 for the single-agent tool path,
and 5/5 for the controlled workflow. This is intentionally reported as a
known-template pilot: it does not establish unfamiliar-query generalization,
report quality, variance, or statistical significance.
